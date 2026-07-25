"""Budget-aware, dataset-independent model training for the competition."""

import math
import time

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

        batch_size = getattr(train_dataloader, "batch_size", None) or 64
        # Linear scaling is useful, but clipping prevents surprising loaders from
        # producing an excessively large learning rate.
        self.learning_rate = min(0.2, max(0.01, 0.05 * batch_size / 128.0))
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

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate, momentum=0.9,
            nesterov=True, weight_decay=wd,
        )

        self._use_amp = self.device.type == "cuda" and torch.cuda.is_available()
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp) # pinned cuda version likely uses old (now deprecated) api
        self._best_state = None
        self._best_accuracy = -1.0

    def _time_left(self):
        try:
            return float(self.clock.check())
        except Exception:
            return float(self.metadata.get("time_remaining", 3600.0))

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
        try:
            return nn.CrossEntropyLoss(label_smoothing=0.1)
        except TypeError:
            return nn.CrossEntropyLoss()

    def _prediction_reserve(self):
        """Keep a conservative tail for restoring weights and prediction."""
        initial = float(self.metadata.get("time_remaining", self._time_left()))
        # Ten percent is useful for short stress tests; cap it so long allocations
        # do not waste hours. The 30 second floor covers loader/model startup.
        return min(300.0, max(30.0, 0.10 * max(0.0, initial)))

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
        self.model.to(self.device)
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
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True).long()
                self._set_learning_rate(global_step, total_steps)
                self.optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=self._use_amp): # pinned cuda version likely uses old (now deprecated) api
                    output = self._logits(self.model(data))
                    loss = self.criterion(output, target)
                self._scaler.scale(loss).backward()
                self._scaler.step(self.optimizer)
                self._scaler.update()

                correct += (output.detach().argmax(1) == target).sum().item()
                seen += target.numel()
                global_step += 1

            epoch_seconds = time.perf_counter() - epoch_start
            if stopped_early:
                break

            valid_accuracy = self.evaluate(reserve)
            train_accuracy = correct / max(1, seen)
            if valid_accuracy is not None and valid_accuracy > self._best_accuracy:
                self._best_accuracy = valid_accuracy
                # CPU checkpoint avoids consuming scarce accelerator memory.
                self._best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
            print("  Epoch {:>3}/{:<3} | train {:>6.2f}% | valid {} | {:.1f}s".format(
                epoch + 1, self.max_epochs, 100.0 * train_accuracy,
                "{:>6.2f}%".format(100.0 * valid_accuracy)
                if valid_accuracy is not None else "skipped", epoch_seconds))

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        self.model.to(self.device)
        print("  Training finished in {:.1f}s; best validation accuracy: {}".format(
            time.perf_counter() - started,
            "{:.2f}%".format(100.0 * self._best_accuracy)
            if self._best_accuracy >= 0 else "unavailable"))

        # =================================================================
        # INCUMBENT CHALLENGE: If time permits, train a runner-up and keep
        # whichever model has better validation accuracy.
        # =================================================================
        self._try_runner_ups(epoch_seconds or 15.0, reserve)

        return self.model

    def _try_runner_ups(self, epoch_seconds, reserve):
        """
        If enough time remains after primary training, train runner-up
        architectures and keep the best (incumbent challenge).
        """
        runner_ups = self.metadata.get('runner_up_configs', [])
        if not runner_ups:
            return

        # Need enough time for at least ~20 epochs of the runner-up + prediction
        min_time_needed = epoch_seconds * 20 + reserve + 60
        time_left = self._time_left()

        if time_left <= min_time_needed:
            print(f"\n  No time for runner-up challenge ({time_left:.0f}s left, "
                  f"need {min_time_needed:.0f}s)")
            return

        incumbent_accuracy = self._best_accuracy
        dropout_rate = self.metadata.get('dropout_rate', 0.1)
        in_channels = self.metadata['input_shape'][1]
        num_classes = self.metadata['num_classes']

        print(f"\n  === Runner-Up Challenge (incumbent: {100*incumbent_accuracy:.2f}%) ===")

        for idx, runner in enumerate(runner_ups):
            # Re-check time before each runner-up
            time_left = self._time_left()
            if time_left <= min_time_needed:
                print(f"  Skipping runner-up {idx+1} (insufficient time)")
                break

            print(f"  Training runner-up {idx+1}: {runner['island']} "
                  f"({runner['params']:,} params)...")

            try:
                challenger = build_model_from_config(
                    runner['cell_config'], in_channels, num_classes,
                    runner['n_cells'], runner['init_channels'], dropout_rate
                )
            except Exception:
                print(f"    Failed to build runner-up, skipping")
                continue

            challenger.to(self.device)
            challenger_acc = self._train_challenger(challenger, reserve)

            if challenger_acc is not None and challenger_acc > incumbent_accuracy:
                print(f"    Runner-up WINS: {100*challenger_acc:.2f}% > "
                      f"{100*incumbent_accuracy:.2f}%")
                self.model = challenger
                self._best_accuracy = challenger_acc
                incumbent_accuracy = challenger_acc
            else:
                acc_str = f"{100*challenger_acc:.2f}%" if challenger_acc else "N/A"
                print(f"    Incumbent holds: {acc_str} <= "
                      f"{100*incumbent_accuracy:.2f}%")
                # Free challenger memory
                del challenger
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def _train_challenger(self, model, reserve):
        """Train a challenger model with remaining time and return its best val accuracy."""
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.learning_rate, momentum=0.9,
            nesterov=True, weight_decay=5e-4,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp)
        best_acc = -1.0
        best_state = None

        batches_per_epoch = max(1, len(self.train_dataloader))
        # Use remaining time minus reserve for training
        max_epochs = min(self.max_epochs, max(1, int(self._time_left() / 30.0)))
        total_steps = max(1, max_epochs * batches_per_epoch)
        global_step = 0

        for epoch in range(max_epochs):
            if self._time_left() <= reserve + 60:
                break

            model.train()
            for data, target in self.train_dataloader:
                if self._time_left() <= reserve + 60:
                    break
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True).long()

                # Cosine LR schedule
                warmup = max(1, int(0.10 * total_steps))
                if global_step < warmup:
                    factor = float(global_step + 1) / warmup
                else:
                    progress = (global_step - warmup) / max(1, total_steps - warmup)
                    factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
                lr = self.learning_rate * factor
                for group in optimizer.param_groups:
                    group["lr"] = lr

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self._use_amp):
                    output = self._logits(model(data))
                    loss = self.criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                global_step += 1

            # Evaluate
            model.eval()
            correct = 0
            seen = 0
            with torch.no_grad():
                for data, target in self.valid_dataloader:
                    if self._time_left() <= reserve:
                        break
                    data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True).long()
                    with torch.cuda.amp.autocast(enabled=self._use_amp):
                        output = self._logits(model(data))
                    correct += (output.argmax(1) == target).sum().item()
                    seen += target.numel()

            if seen > 0:
                val_acc = correct / seen
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}

        # Restore best state
        if best_state is not None:
            model.load_state_dict(best_state)
        model.to(self.device)

        return best_acc if best_acc >= 0 else None

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
        with torch.no_grad():
            for batch in test_loader:
                # Be tolerant of a test dataset that returns (image,) rather than
                # a bare tensor, while preserving sample order.
                data = batch[0] if isinstance(batch, (tuple, list)) else batch
                data = data.to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=self._use_amp):
                    output = self._logits(self.model(data))
                predictions.extend(output.argmax(1).cpu().tolist())
        return predictions
