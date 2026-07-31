"""Budget-aware, dataset-independent model training for the competition."""

import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from helpers import build_model_from_config


class Trainer:
    """Train the model while leaving enough time for test-set prediction.

    The policy intentionally uses only information available for every hidden
    dataset.  In particular, the clock is checked during epochs; an epoch count
    computed once from ``metadata['time_remaining']`` can otherwise become stale.
    """

    def __init__(self, model, device, train_dataloader, valid_dataloader,
                 metadata, clock):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.metadata = metadata
        self.clock = clock
        self.bo_config = dict(metadata.get("bo_config", {}))
        self.master_seed = int(metadata.get("seed", 42))
        self.initial_budget = float(
            metadata.get("time_remaining", self._time_left()))
        self.primary_seed = int(
            metadata.get("primary_candidate", {}).get(
                "seed", self.master_seed))

        batch_size = getattr(train_dataloader, "batch_size", None) or 64
        # Linear scaling is useful, but clipping prevents surprising loaders from
        # producing an excessively large learning rate.
        self.learning_rate = min(0.2, max(0.01, 0.05 * batch_size / 128.0))
        if "learning_rate" in self.bo_config:
            self.learning_rate = float(self.bo_config["learning_rate"])
        self.max_epochs = self._epoch_cap()
        self.criterion = self._make_criterion()

        # (E) Adaptive weight decay: stronger for simple tasks to reduce overfitting,
        # standard for complex tasks to allow capacity
        n_samples = metadata.get('input_shape', [50000])[0]
        num_classes = metadata.get('num_classes', 10)
        spatial_size = 1
        if len(metadata.get('input_shape', [])) >= 4:
            spatial_size = metadata['input_shape'][2] * metadata['input_shape'][3]

        if num_classes <= 10 and spatial_size <= 512:
            wd = 2e-3  # strong regularization for simple/overfit-prone tasks
        else:
            wd = 5e-4  # standard
        if "weight_decay" in self.bo_config:
            wd = float(self.bo_config["weight_decay"])
        self.weight_decay = wd

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate, momentum=0.9,
            nesterov=True, weight_decay=wd,
        )

        self._use_amp = self.device.type == "cuda" and torch.cuda.is_available()
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp) # pinned cuda version likely uses old (now deprecated) api
        self._best_state = None
        self._best_accuracy = -1.0
        self._epochs_without_improvement = 0
        self._ensemble_models = []
        self._ensemble_weights = [1.0]
        self._model_records = []
        self._incumbent_model = self.model
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        safe_codename = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in str(metadata.get("codename", "dataset")))
        self._checkpoint_path = os.path.join(
            "predictions", "{}_best.pt".format(safe_codename))

    def _time_left(self):
        try:
            remaining = float(self.clock.check())
        except Exception:
            remaining = float(
                self.metadata.get("time_remaining", 3600.0))
        deadline = self.metadata.get("dataset_deadline")
        if deadline is not None:
            remaining = min(
                remaining, float(deadline) - time.perf_counter())
        return remaining

    def _reset_randomness(self, seed):
        """Make candidate initialization, shuffling and augmentation repeatable."""
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        generator = getattr(self.train_dataloader, "generator", None)
        if generator is not None:
            generator.manual_seed(seed)
        dataset_generator = getattr(
            getattr(self.train_dataloader, "dataset", None), "generator", None)
        if dataset_generator is not None:
            dataset_generator.manual_seed(seed + 1)

    def _epoch_cap(self):
        """A coarse cap; measured epoch time supplies the real stopping rule."""
        remaining = max(0.0, self._time_left())
        n = len(getattr(self.train_dataloader, "dataset", ()))
        # Allow more epochs across the board — the wall-clock check is the
        # real safety net. These caps just prevent extreme outliers.
        size_cap = 200 if n < 5_000 else (150 if n < 25_000 else 120)
        time_cap = max(1, int(remaining / 20.0))
        return min(size_cap, time_cap)

    def _make_criterion(self):
        # Mild label smoothing is a robust regularizer on unknown image tasks.
        # Retain compatibility with older PyTorch installations.
        diagnostics = self.metadata.get("diagnostics", {})
        smoothing = float(self.bo_config.get(
            "label_smoothing",
            0.05 if diagnostics.get("encoded_likely") else 0.1))
        weights = None
        labels = getattr(getattr(self.train_dataloader, "dataset", None),
                         "y", None)
        weighting = self.bo_config.get("class_weighting", "inverse")
        threshold = float(self.bo_config.get("class_weighting_threshold", 3.0))
        if (labels is not None and weighting != "none" and
                diagnostics.get("class_imbalance_ratio", 1.0) >= threshold):
            labels = torch.as_tensor(labels, dtype=torch.long)
            counts = torch.bincount(
                labels, minlength=int(self.metadata["num_classes"])).float()
            weights = counts.sum() / counts.clamp_min(1.0)
            if weighting == "inverse_sqrt":
                weights = weights.sqrt()
            elif weighting != "inverse":
                raise ValueError(
                    "unsupported class weighting mode: {}".format(weighting))
            weights = weights / weights.mean()
        try:
            return nn.CrossEntropyLoss(
                weight=weights, label_smoothing=smoothing)
        except TypeError:
            return nn.CrossEntropyLoss(weight=weights)

    def _prediction_reserve(self):
        """Keep a conservative tail for restoring weights and prediction."""
        initial = float(self.metadata.get("time_remaining", self._time_left()))
        # Ten percent is useful for short stress tests; cap it so long allocations
        # do not waste hours. The 30 second floor covers loader/model startup.
        test_size = int(self.metadata.get("test_size", 0))
        return min(600.0, max(
            45.0, 0.10 * max(0.0, initial), 30.0 + 0.003 * test_size))

    def _save_checkpoint(self):
        """Persist the incumbent as a second line of defence against failures."""
        if self.metadata.get("disable_checkpoint", False):
            return
        try:
            directory = os.path.dirname(self._checkpoint_path)
            if directory and not os.path.isdir(directory):
                os.makedirs(directory)
            torch.save(self._best_state, self._checkpoint_path)
        except Exception as error:
            print("  Checkpoint warning: {}".format(error))

    def _set_learning_rate(self, step, total_steps):
        """Ten-percent warm-up followed by per-step cosine decay.
        Longer warmup helps with initially unstable gradients on novel datasets."""
        warmup = max(1, int(0.10 * total_steps))
        if step < warmup:
            factor = float(step + 1) / warmup
        else:
            progress = (step - warmup) / max(1, total_steps - warmup)
            factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        lr = self.learning_rate * factor
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    @staticmethod
    def _logits(output):
        # Accommodate common NAS models returning auxiliary values.
        if isinstance(output, (tuple, list)):
            return output[0]
        if isinstance(output, dict):
            return output.get("logits", next(iter(output.values())))
        return output

    def train(self):
        self._reset_randomness(self.primary_seed)
        self.model.to(self.device)
        self.criterion.to(self.device)
        reserve = self._prediction_reserve()
        batches_per_epoch = max(1, len(self.train_dataloader))
        total_steps = max(1, self.max_epochs * batches_per_epoch)
        global_step = 0
        epoch_seconds = None
        started = time.perf_counter()

        print("  Trainer: SGD+nesterov, cosine decay, AMP={}, epoch cap={}, "
              "prediction reserve={:.0f}s".format(
                  self._use_amp, self.max_epochs, reserve))

        for epoch in range(self.max_epochs):
            # Once one epoch is measured, do not begin another that is unlikely
            # to leave time for validation and prediction.
            margin = reserve + (1.25 * epoch_seconds if epoch_seconds else 0.0)
            if self._time_left() <= margin:
                break

            epoch_start = time.perf_counter()
            self.model.train()
            correct = 0
            seen = 0
            stopped_early = False

            for data, target in self.train_dataloader:
                if self._time_left() <= reserve:
                    stopped_early = True
                    break
                self._set_learning_rate(global_step, total_steps)
                batch_correct, batch_seen = self._train_batch(data, target)
                correct += batch_correct
                seen += batch_seen
                global_step += 1

            epoch_seconds = time.perf_counter() - epoch_start
            if stopped_early:
                break

            valid_accuracy = self.evaluate(reserve)
            train_accuracy = correct / max(1, seen)
            if valid_accuracy is not None and valid_accuracy > self._best_accuracy:
                self._best_accuracy = valid_accuracy
                self._epochs_without_improvement = 0
                # CPU checkpoint avoids consuming scarce accelerator memory.
                self._best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                self._save_checkpoint()
            elif valid_accuracy is not None:
                self._epochs_without_improvement += 1
            print("  Epoch {:>3}/{:<3} | train {:>6.2f}% | valid {} | {:.1f}s".format(
                epoch + 1, self.max_epochs, 100.0 * train_accuracy,
                "{:>6.2f}%".format(100.0 * valid_accuracy)
                if valid_accuracy is not None else "skipped", epoch_seconds))
            patience = 30
            if (epoch >= 30 and
                    self._epochs_without_improvement >= patience):
                print("  Early stopping after {} stale epochs".format(
                    self._epochs_without_improvement))
                break

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        self.model.to(self.device)
        print("  Training finished in {:.1f}s; best validation accuracy: {}".format(
            time.perf_counter() - started,
            "{:.2f}%".format(100.0 * self._best_accuracy)
            if self._best_accuracy >= 0 else "unavailable"))

        self._model_records = [{
            "model": self.model,
            "accuracy": self._best_accuracy,
            "island": self.metadata.get(
                "primary_candidate", {}).get("island", "primary"),
            "seed": self.primary_seed,
        }]

        # A safe incumbent now exists. Allocate the rest of the useful budget
        # to a progressively narrowed, resumable candidate portfolio.
        try:
            self._run_portfolio(epoch_seconds or 15.0, reserve)
        except Exception as error:
            print("  Portfolio failure contained; restoring incumbent: {}".format(
                error))
            self.model = self._incumbent_model
            if self._best_state is not None:
                self.model.load_state_dict(self._best_state)
            self.model.to(self.device)
        try:
            self._select_validated_ensemble(reserve)
        except Exception as error:
            print("  Ensemble validation failed safely: {}".format(error))
            self._ensemble_models = []
            self._ensemble_weights = [1.0]
            self.model = self._incumbent_model
            if self._best_state is not None:
                self.model.load_state_dict(self._best_state)
            self.model.to(self.device)

        return self.model

    def _train_batch(self, cpu_data, cpu_target):
        """Train with automatic microbatch fallback after accelerator OOM."""
        microbatch = len(cpu_data)
        while microbatch >= 1:
            self.optimizer.zero_grad(set_to_none=True)
            correct = seen = 0
            try:
                for start in range(0, len(cpu_data), microbatch):
                    data = cpu_data[start:start + microbatch].to(
                        self.device, non_blocking=True)
                    target = cpu_target[start:start + microbatch].to(
                        self.device, non_blocking=True).long()
                    with torch.cuda.amp.autocast(enabled=self._use_amp):
                        output = self._logits(self.model(data))
                        # Preserve the full-batch gradient scale.
                        loss = self.criterion(output, target)
                        loss = loss * (target.numel() / float(len(cpu_data)))
                    self._scaler.scale(loss).backward()
                    correct += (
                        output.detach().argmax(1) == target).sum().item()
                    seen += target.numel()
                self._scaler.step(self.optimizer)
                self._scaler.update()
                return correct, seen
            except RuntimeError as error:
                if ("out of memory" not in str(error).lower() or
                        microbatch == 1):
                    raise
                self.optimizer.zero_grad(set_to_none=True)
                microbatch = max(1, microbatch // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("  OOM recovery: training microbatch={}".format(
                    microbatch))
        return 0, 0

    def _portfolio_guard(self, reserve, epoch_estimate):
        """Leave room for final validation, model restore and prediction."""
        return self._time_left() > (
            reserve + 120.0 + max(45.0, 1.5 * float(epoch_estimate)))

    def _new_candidate_state(self, config, default_epoch_seconds):
        return {
            "config": dict(config),
            "model": None,
            "epochs": 0,
            "best_accuracy": -1.0,
            "best_state": None,
            "stale_epochs": 0,
            "epoch_seconds": float(default_epoch_seconds),
            "stopped": False,
            "failed": False,
            "record": None,
        }

    def _build_candidate(self, state):
        config = state["config"]
        seed = int(config.get("seed", self.master_seed))
        self._reset_randomness(seed)
        dropout = float(config.get(
            "dropout_rate", self.metadata.get("dropout_rate", 0.1)))
        model = build_model_from_config(
            config["cell_config"],
            self.metadata["input_shape"][1],
            self.metadata["num_classes"],
            config["n_cells"],
            config["init_channels"],
            dropout,
            **dict(config.get("model_kwargs", {})),
        )
        state["model"] = model.cpu()

    def _register_candidate(self, state):
        if state["best_state"] is None:
            return
        state["model"].load_state_dict(state["best_state"])
        record = state.get("record")
        if record is None:
            record = {
                "model": state["model"],
                "accuracy": state["best_accuracy"],
                "island": state["config"].get("island", "challenger"),
                "seed": int(state["config"].get("seed", self.master_seed)),
            }
            state["record"] = record
            self._model_records.append(record)
        else:
            record["accuracy"] = state["best_accuracy"]

        if state["best_accuracy"] > self._best_accuracy:
            previous = self._best_accuracy
            self._best_accuracy = state["best_accuracy"]
            self._best_state = {
                key: value.detach().cpu().clone()
                for key, value in state["best_state"].items()
            }
            self._incumbent_model = state["model"]
            self._save_checkpoint()
            print("    New incumbent: {:.2f}% > {:.2f}% ({})".format(
                100.0 * self._best_accuracy,
                100.0 * previous,
                state["config"].get("island", "challenger"),
            ))

    def _train_candidate_to(self, state, target_epochs, reserve):
        """Continue one candidate to a fidelity target and retain its best state."""
        if state["failed"] or state["stopped"] or state["epochs"] >= target_epochs:
            return
        if state["model"] is None:
            self._build_candidate(state)

        config = state["config"]
        seed = int(config.get("seed", self.master_seed))
        self._reset_randomness(seed + state["epochs"] * 1009)
        model = state["model"].to(self.device)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.learning_rate, momentum=0.9,
            nesterov=True, weight_decay=self.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp)
        batches_per_epoch = max(1, len(self.train_dataloader))
        portfolio_cap = max(self.max_epochs, int(target_epochs))
        total_steps = max(1, portfolio_cap * batches_per_epoch)
        global_step = state["epochs"] * batches_per_epoch

        while state["epochs"] < target_epochs and not state["stopped"]:
            if not self._portfolio_guard(reserve, state["epoch_seconds"]):
                break
            epoch_started = time.perf_counter()
            model.train()
            completed_epoch = True
            try:
                for data, target in self.train_dataloader:
                    if self._time_left() <= reserve + 120:
                        completed_epoch = False
                        break
                    warmup = max(1, int(0.10 * total_steps))
                    if global_step < warmup:
                        factor = float(global_step + 1) / warmup
                    else:
                        progress = (
                            (global_step - warmup) /
                            max(1, total_steps - warmup))
                        factor = 0.5 * (
                            1.0 + math.cos(
                                math.pi * min(1.0, progress)))
                    for group in optimizer.param_groups:
                        group["lr"] = self.learning_rate * factor
                    data = data.to(self.device, non_blocking=True)
                    target = target.to(
                        self.device, non_blocking=True).long()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=self._use_amp):
                        output = self._logits(model(data))
                        loss = self.criterion(output, target)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    global_step += 1
            except RuntimeError as error:
                if "out of memory" in str(error).lower():
                    print("    Challenger OOM; retiring {}".format(
                        config.get("island", "challenger")))
                    state["failed"] = True
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    break
                raise

            state["epoch_seconds"] = max(
                0.1, time.perf_counter() - epoch_started)
            if not completed_epoch:
                break
            state["epochs"] += 1
            accuracy = self._evaluate_model(model, reserve + 90)
            if accuracy is None:
                break
            if accuracy > state["best_accuracy"]:
                state["best_accuracy"] = accuracy
                state["best_state"] = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                state["stale_epochs"] = 0
                self._register_candidate(state)
            else:
                state["stale_epochs"] += 1
            if (state["epochs"] >= 30 and
                    state["stale_epochs"] >= 30):
                state["stopped"] = True

        if state["best_state"] is not None:
            model.load_state_dict(state["best_state"])
        state["model"] = model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._register_candidate(state)
        print("    {:<22} seed={:<10} epochs={:<3} best={}".format(
            config.get("island", "challenger"),
            int(config.get("seed", self.master_seed)),
            state["epochs"],
            "{:.2f}%".format(100.0 * state["best_accuracy"])
            if state["best_accuracy"] >= 0 else "unavailable",
        ))

    @staticmethod
    def _promote(states, count):
        ranked = sorted(
            [state for state in states
             if state["best_accuracy"] >= 0 and not state["failed"]],
            key=lambda state: state["best_accuracy"],
            reverse=True,
        )
        promoted = ranked[:max(1, min(count, len(ranked)))]
        # Give a detected specialist enough fidelity to prove itself, while it
        # still cannot displace a baseline incumbent merely by existing.
        specialists = [
            state for state in ranked
            if state["config"].get("specialist", False)]
        if specialists and specialists[0] not in promoted:
            promoted.append(specialists[0])
        return promoted

    def _run_portfolio(self, primary_epoch_seconds, reserve):
        configs = list(self.metadata.get("runner_up_configs", []))
        if not configs:
            return
        # Longer final-round allocations can afford additional independent
        # initializations. They are derived deterministically and never replace
        # the original baseline portfolio.
        if self.initial_budget > 14_400:
            repeat_sources = [
                config for config in configs
                if not config.get("specialist", False)
            ][:3]
            for repeat_index, source in enumerate(repeat_sources, 2):
                repeated = dict(source)
                repeated["model_kwargs"] = dict(
                    source.get("model_kwargs", {}))
                repeated["seed"] = (
                    int(source.get("seed", self.master_seed)) +
                    repeat_index * 1_000_003) & 0x7fffffff
                repeated["island"] = "{}-repeat{}".format(
                    source.get("island", "candidate"), repeat_index)
                configs.append(repeated)
        if not self._portfolio_guard(reserve, primary_epoch_seconds):
            print("\n  Portfolio race skipped: prediction safety margin reached")
            return

        print("\n  === Anytime Portfolio Race: {} challengers ===".format(
            len(configs)))
        # The primary is now a disk-backed incumbent, so move it off the GPU
        # while challengers race.
        self.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        states = [
            self._new_candidate_state(config, primary_epoch_seconds)
            for config in configs
        ]
        active = states
        portfolio_epoch_cap = (
            max(self.max_epochs, 240)
            if self.initial_budget > 14_400 else self.max_epochs
        )
        rung_targets = [5, 20, portfolio_epoch_cap]
        for rung_index, target in enumerate(rung_targets, 1):
            if not active or not self._portfolio_guard(
                    reserve, primary_epoch_seconds):
                break
            print("  Rung {}: {} candidates -> {} epochs".format(
                rung_index, len(active), target))
            # Earlier rungs are cheap and synchronous. The final rung rotates
            # through ten-epoch blocks so one slow model cannot monopolize the
            # remaining allocation.
            final_rung = rung_index == len(rung_targets)
            while active:
                progressed = False
                for state in list(active):
                    if not self._portfolio_guard(
                            reserve, state["epoch_seconds"]):
                        break
                    state_target = (
                        min(target, state["epochs"] + 10)
                        if final_rung else target)
                    before = state["epochs"]
                    try:
                        self._train_candidate_to(
                            state, state_target, reserve)
                    except Exception as error:
                        state["failed"] = True
                        print("    Challenger failed safely ({}): {}".format(
                            state["config"].get("island", "challenger"),
                            error))
                        if state.get("best_state") is not None:
                            state["model"].load_state_dict(
                                state["best_state"])
                            state["model"].cpu()
                            self._register_candidate(state)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    progressed = progressed or state["epochs"] > before
                if (not final_rung or not progressed or
                        all(state["failed"] or state["stopped"] or
                            state["epochs"] >= target for state in active)):
                    break
            if rung_index < len(rung_targets):
                active = self._promote(
                    active, int(math.ceil(len(active) / 2.0)))

        # Successive halving prioritizes promising candidates. If useful time
        # remains, revisit the rest in validation order instead of terminating.
        backlog = sorted(
            [state for state in states
             if not state["failed"] and not state["stopped"] and
             state["epochs"] < portfolio_epoch_cap],
            key=lambda state: state["best_accuracy"],
            reverse=True,
        )
        for state in backlog:
            if not self._portfolio_guard(
                    reserve, state["epoch_seconds"]):
                break
            try:
                self._train_candidate_to(
                    state, portfolio_epoch_cap, reserve)
            except Exception as error:
                state["failed"] = True
                print("    Backlog candidate failed safely ({}): {}".format(
                    state["config"].get("island", "challenger"),
                    error))
                if state.get("best_state") is not None:
                    state["model"].load_state_dict(state["best_state"])
                    state["model"].cpu()
                    self._register_candidate(state)

        valid_records = [
            record for record in self._model_records
            if record.get("accuracy", -1.0) >= 0]
        if valid_records:
            winner = max(valid_records, key=lambda record: record["accuracy"])
            self.model = winner["model"]
            self._best_accuracy = winner["accuracy"]
            self._best_state = {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            }
            self._incumbent_model = self.model
            self._save_checkpoint()
            print("  Portfolio incumbent: {} at {:.2f}%".format(
                winner["island"], 100.0 * winner["accuracy"]))
        self.model.to(self.device)

    def _evaluate_model(self, model, reserve=0.0):
        model.eval()
        correct = 0
        seen = 0
        with torch.no_grad():
            for data, target in self.valid_dataloader:
                if self._time_left() <= reserve:
                    return None
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True).long()
                with torch.cuda.amp.autocast(enabled=self._use_amp):
                    output = self._logits(model(data))
                correct += (output.argmax(1) == target).sum().item()
                seen += target.numel()
        return correct / float(seen) if seen else None

    def _validation_logits(self, model, reserve):
        if self._time_left() <= reserve + 30:
            return None
        model.to(self.device)
        model.eval()
        logits = []
        targets = []
        with torch.no_grad():
            for data, target in self.valid_dataloader:
                if self._time_left() <= reserve + 30:
                    model.cpu()
                    return None
                data = data.to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=self._use_amp):
                    output = self._logits(model(data))
                logits.append(output.detach().float().cpu())
                targets.append(target.detach().long().cpu())
        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not logits:
            return None
        return torch.cat(logits), torch.cat(targets)

    @staticmethod
    def _stratified_half_masks(targets):
        first = torch.zeros(len(targets), dtype=torch.bool)
        class_counts = {}
        for index, label in enumerate(targets.tolist()):
            count = class_counts.get(label, 0)
            first[index] = (count % 2 == 0)
            class_counts[label] = count + 1
        return first, ~first

    def _select_validated_ensemble(self, reserve):
        """Select a small, robustly validated weighted ensemble."""
        self._ensemble_models = []
        self._ensemble_weights = [1.0]
        records = sorted(
            [record for record in self._model_records
             if record.get("accuracy", -1.0) >= 0],
            key=lambda record: record["accuracy"],
            reverse=True,
        )
        if len(records) < 2 or self._time_left() <= reserve + 90:
            return
        winner = records[0]
        winner_result = self._validation_logits(winner["model"], reserve)
        if winner_result is None:
            return
        winner_logits, targets = winner_result
        winner_predictions = winner_logits.argmax(1)
        masks = self._stratified_half_masks(targets)
        winner_full = float((winner_predictions == targets).float().mean())
        winner_halves = [
            float((winner_predictions[mask] == targets[mask]).float().mean())
            for mask in masks if bool(mask.any())
        ]

        # Prefer one strong representative per island, then fill remaining
        # slots by validation accuracy. A five-point cutoff still includes the
        # complementary Gutenberg specialist/CNN pairing observed during
        # development.
        eligible = [
            record for record in records[1:]
            if record["accuracy"] >= winner["accuracy"] - 0.05
        ]
        shortlist = []
        used_islands = {winner.get("island")}
        for record in eligible:
            island = record.get("island")
            if island not in used_islands:
                shortlist.append(record)
                used_islands.add(island)
            if len(shortlist) == 4:
                break
        for record in eligible:
            if len(shortlist) == 4:
                break
            if all(record is not selected for selected in shortlist):
                shortlist.append(record)

        best_partner = None
        best_weight = None
        best_accuracy = winner_full
        best_halves = winner_halves
        cached_logits = {}
        incumbent_weights = (0.8, 0.7, 0.6, 0.5)
        for partner in shortlist:
            if self._time_left() <= reserve + 45:
                break
            partner_result = self._validation_logits(
                partner["model"], reserve)
            if partner_result is None:
                break
            partner_logits, partner_targets = partner_result
            if (len(partner_targets) != len(targets) or
                    not torch.equal(partner_targets, targets)):
                continue
            # Identical predictions provide no useful robustness diversity.
            disagreement = float(
                (partner_logits.argmax(1) != winner_predictions)
                .float().mean())
            if disagreement < 0.001:
                continue
            cached_logits[id(partner)] = partner_logits
            for incumbent_weight in incumbent_weights:
                combined = (
                    incumbent_weight * winner_logits +
                    (1.0 - incumbent_weight) * partner_logits
                )
                predictions = combined.argmax(1)
                full_accuracy = float(
                    (predictions == targets).float().mean())
                half_accuracies = [
                    float((predictions[mask] == targets[mask]).float().mean())
                    for mask in masks if bool(mask.any())
                ]
                robust_halves = all(
                    ensemble_half >= winner_half - 0.0005
                    for ensemble_half, winner_half
                    in zip(half_accuracies, winner_halves))
                if (full_accuracy >= winner_full + 0.0015 and
                        robust_halves and full_accuracy > best_accuracy):
                    best_accuracy = full_accuracy
                    best_halves = half_accuracies
                    best_partner = partner
                    best_weight = incumbent_weight

        self.model = winner["model"].to(self.device)
        self._incumbent_model = self.model
        self._best_accuracy = winner["accuracy"]
        if best_partner is not None:
            best_partner["model"].eval()
            self._ensemble_models = [best_partner["model"]]
            self._ensemble_weights = [
                best_weight, 1.0 - best_weight]

            # A third model gets one deliberately fixed, incumbent-heavy test.
            # It must improve on the accepted pair and pass the same split gate.
            third_record = None
            third_accuracy = best_accuracy
            for candidate in shortlist:
                if (candidate is best_partner or
                        id(candidate) not in cached_logits):
                    continue
                if self._time_left() <= reserve + 30:
                    break
                combined = (
                    0.5 * winner_logits +
                    0.3 * cached_logits[id(best_partner)] +
                    0.2 * cached_logits[id(candidate)]
                )
                predictions = combined.argmax(1)
                full_accuracy = float(
                    (predictions == targets).float().mean())
                half_accuracies = [
                    float((predictions[mask] == targets[mask]).float().mean())
                    for mask in masks if bool(mask.any())
                ]
                robust_halves = all(
                    ensemble_half >= pair_half - 0.0005
                    for ensemble_half, pair_half
                    in zip(half_accuracies, best_halves))
                if (full_accuracy >= best_accuracy + 0.0015 and
                        robust_halves and full_accuracy > third_accuracy):
                    third_record = candidate
                    third_accuracy = full_accuracy

            if third_record is not None:
                third_record["model"].eval()
                self._ensemble_models.append(third_record["model"])
                self._ensemble_weights = [0.5, 0.3, 0.2]
                print(
                    "  Validated 3-model ensemble: {:.2f}% -> {:.2f}% "
                    "with {} + {}".format(
                        100.0 * winner_full,
                        100.0 * third_accuracy,
                        best_partner["island"],
                        third_record["island"],
                    ))
            else:
                print("  Validated weighted ensemble: {:.2f}% -> {:.2f}% "
                      "with {} ({:.0f}/{:.0f})".format(
                          100.0 * winner_full,
                          100.0 * best_accuracy,
                          best_partner["island"],
                          100.0 * best_weight,
                          100.0 * (1.0 - best_weight),
                      ))
        else:
            print("  Ensemble rejected: no robust validation improvement")

    def evaluate(self, reserve=0.0):
        self.model.eval()
        correct = 0
        seen = 0
        with torch.no_grad():
            for data, target in self.valid_dataloader:
                if self._time_left() <= reserve:
                    return None
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True).long()
                with torch.cuda.amp.autocast(enabled=self._use_amp):
                    output = self._logits(self.model(data))
                correct += (output.argmax(1) == target).sum().item()
                seen += target.numel()
        return correct / max(1, seen)

    def predict(self, test_loader):
        self.model.to(self.device)
        self.model.eval()
        predictions = []
        models = [self.model] + list(self._ensemble_models)
        weights = list(self._ensemble_weights)
        if len(weights) != len(models):
            weights = [1.0 / len(models)] * len(models)
        try:
            for candidate in models:
                candidate.to(self.device)
                candidate.eval()
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            print("  Prediction safety: ensemble did not fit; using incumbent")
            for extra in self._ensemble_models:
                extra.cpu()
            self._ensemble_models = []
            self._ensemble_weights = [1.0]
            models = [self.model]
            weights = [1.0]
            self.model.to(self.device)
            self.model.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        started = time.perf_counter()
        processed = 0
        with torch.no_grad():
            for batch in test_loader:
                # Be tolerant of a test dataset that returns (image,) rather than
                # a bare tensor, while preserving sample order.
                data = batch[0] if isinstance(batch, (tuple, list)) else batch
                try:
                    data = data.to(self.device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=self._use_amp):
                        logits = [self._logits(candidate(data))
                                  for candidate in models]
                        output = self._weighted_logits(logits, weights)
                except RuntimeError as error:
                    if ("out of memory" not in str(error).lower() or
                            len(data) <= 1):
                        raise
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    output = self._predict_in_chunks(
                        models, weights, data.cpu())
                predicted_indices = output.argmax(1).cpu().tolist()
                label_values = self.metadata.get("label_values")
                if label_values is None:
                    predictions.extend(predicted_indices)
                else:
                    predictions.extend(
                        label_values[index] for index in predicted_indices)
                processed += len(data)
                # If the ensemble makes the measured prediction estimate unsafe,
                # discard it for all remaining batches.
                elapsed = time.perf_counter() - started
                if (len(models) > 1 and processed > 0 and
                        self._time_left() <
                        1.5 * elapsed * max(
                            0.0, self.metadata.get("test_size", processed) /
                            float(processed) - 1.0) + 30.0):
                    models = models[:1]
                    weights = [1.0]
                    print("  Prediction safety: disabling ensemble")
        return predictions

    @staticmethod
    def _weighted_logits(logits, weights):
        return sum(
            weight * output for weight, output in zip(weights, logits))

    def _predict_in_chunks(self, models, weights, cpu_data):
        """Recursively reduce prediction microbatch size after an OOM."""
        if len(cpu_data) <= 1:
            data = cpu_data.to(self.device)
            logits = [self._logits(model(data)) for model in models]
            return self._weighted_logits(logits, weights)
        midpoint = len(cpu_data) // 2
        outputs = []
        for chunk in (cpu_data[:midpoint], cpu_data[midpoint:]):
            try:
                data = chunk.to(self.device)
                with torch.cuda.amp.autocast(enabled=self._use_amp):
                    logits = [self._logits(model(data)) for model in models]
                    outputs.append(self._weighted_logits(logits, weights))
            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                outputs.append(self._predict_in_chunks(
                    models, weights, chunk))
        return torch.cat(outputs, dim=0)
