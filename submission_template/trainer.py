"""Budget-aware, dataset-independent model training for the competition."""

import math
import time

import torch
import torch.nn as nn


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
        return self.model

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
