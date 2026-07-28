"""
nas.py - Neural Architecture Search using training-free proxies (NASWOT + SynFlow),
a cell-based search space, diversity islands, and budget-aware candidate selection.
"""

import json
import math
import os
import time
import random

import torch
import torch.nn as nn

from helpers import (
    show_time,
    sample_cell_config,
    build_model_from_config,
    compute_naswot_score,
    compute_synflow_score,
    compute_param_count,
    robust_normalize,
    compute_combined_score,
    assign_island,
    architecture_descriptor,
    diversity_summary,
    select_diverse_candidates,
)


class NAS:
    """
    ====================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    """
    def __init__(self, train_loader, valid_loader, metadata, clock):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.clock = clock

        # Dataset properties
        self.num_classes = metadata['num_classes']
        self.input_shape = metadata['input_shape']  # [n_total, c, h, w]
        self.in_channels = self.input_shape[1]
        self.img_height = self.input_shape[2]
        self.img_width = self.input_shape[3]
        self.time_remaining = metadata.get('time_remaining', 3600)

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Search space parameters — adapted to dataset complexity
        self._configure_search_space()
        bo_config = metadata.get("bo_config", {})
        if "n_cells" in bo_config:
            self.cell_counts = [int(bo_config["n_cells"])]
        if "init_channels" in bo_config:
            self.init_channels_options = [int(bo_config["init_channels"])]
        if "dropout" in bo_config:
            self.dropout_rate = float(bo_config["dropout"])
        self.proxy_weights = self._load_proxy_weights()

    def _time_left(self):
        try:
            return float(self.clock.check())
        except Exception:
            return float(self.metadata.get("time_remaining", 3600.0))

    def _load_proxy_weights(self):
        """Use calibration output when it has been copied into the submission."""
        weights = {"naswot": 0.5, "synflow": 0.5}
        path = os.path.join(os.path.dirname(__file__), "proxy_weights.json")
        try:
            with open(path, "r") as handle:
                loaded = json.load(handle)
            naswot = float(loaded.get("naswot_weight", 0.5))
            if 0.0 <= naswot <= 1.0:
                weights = {"naswot": naswot, "synflow": 1.0 - naswot}
        except Exception:
            pass
        return weights

    def _configure_search_space(self):
        """
        Configure search space parameters based on dataset metadata.
        Key insight: only constrain model size for truly simple tasks that are
        prone to overfitting. For complex tasks, allow large models.
        """
        diagnostics = self.metadata.get("diagnostics", {})
        n_datapoints = diagnostics.get("n_train", self.input_shape[0])
        spatial_size = self.img_height * self.img_width

        # Estimate dataset complexity
        imbalance = diagnostics.get("class_imbalance_ratio", 1.0)
        self.is_simple_task = (
            self.num_classes <= 10 and spatial_size <= 512 and
            n_datapoints < 25000
        )
        self.is_large_spatial = (spatial_size > 1024)

        # Nodes per cell: 3-5 depending on complexity
        if self.num_classes <= 10 and spatial_size <= 1024:
            self.n_nodes = 3
        elif self.num_classes <= 50:
            self.n_nodes = 4
        else:
            self.n_nodes = 5

        # Search space sizing
        if self.is_simple_task:
            self.cell_counts = [2, 3]
            self.init_channels_options = [16, 24]
            self.max_params = 750_000 if n_datapoints >= 5000 else 300_000
            self.dropout_rate = 0.35 if n_datapoints < 5000 else 0.20
        elif spatial_size <= 1024:
            self.cell_counts = [3, 4, 5]
            self.init_channels_options = [32, 48, 64]
            self.max_params = 15_000_000
            self.dropout_rate = 0.1
        else:
            self.cell_counts = [3, 4, 5, 6]
            self.init_channels_options = [32, 48, 64]
            self.max_params = 15_000_000
            self.dropout_rate = 0.1
        if imbalance >= 10.0:
            self.dropout_rate = min(0.4, self.dropout_rate + 0.1)

        # Budget-aware candidate count
        if self.time_remaining > 18000:
            self.n_candidates = 80
        elif self.time_remaining > 7200:
            self.n_candidates = 50
        elif self.time_remaining > 3600:
            self.n_candidates = 40 if self.is_large_spatial else 30
        else:
            self.n_candidates = 25

    """
    ====================================================================================================================
    SEARCH =============================================================================================================
    ====================================================================================================================
    The search function performs:
    1. Sample candidate architectures from the cell-based search space
    2. Score each candidate using training-free proxies (NASWOT + SynFlow)
    3. Group top candidates into diversity islands
    4. Select the best island champion by combined proxy score
    5. Short validation warm-up if time permits
    """
    def search(self):
        print(f"  NAS Search | Device: {self.device} | Candidates: {self.n_candidates}")
        print(f"  Dataset: {self.in_channels}ch, {self.img_height}x{self.img_width}, "
              f"{self.num_classes} classes | Simple: {self.is_simple_task}")
        print(f"  Search space: {self.n_nodes} nodes/cell, "
              f"cells={self.cell_counts}, channels={self.init_channels_options}")
        print(f"  Max params: {self.max_params:,}")

        search_start = time.perf_counter()
        search_deadline = max(15.0, min(
            self.time_remaining * 0.12,
            max(15.0, self._time_left() - self._prediction_reserve() - 60.0)))
        rng = random.Random(42)

        # ==================================================================
        # PHASE 1: Sample candidates and score with training-free proxies
        # ==================================================================
        candidates = []
        print(f"\n  Phase 1: Sampling and scoring {self.n_candidates} candidates...")

        for i in range(self.n_candidates):
            elapsed = time.perf_counter() - search_start
            if elapsed > search_deadline or self._time_left() <= self._prediction_reserve() + 60:
                print(f"  Time budget reached after {i} candidates ({show_time(elapsed)})")
                break

            cell_config = sample_cell_config(self.n_nodes, rng)
            n_cells = rng.choice(self.cell_counts)
            init_channels = rng.choice(self.init_channels_options)

            try:
                model = build_model_from_config(
                    cell_config, self.in_channels, self.num_classes,
                    n_cells, init_channels, self.dropout_rate
                )
            except Exception:
                continue

            param_count = compute_param_count(model)
            if param_count > self.max_params:
                continue

            try:
                naswot_score = compute_naswot_score(
                    model, self.train_loader, self.device, max_samples=24)
                synflow_score = compute_synflow_score(
                    model, self.train_loader, self.device)
            except RuntimeError as error:
                if "out of memory" in str(error).lower():
                    print("  Candidate OOM; reducing search model size")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.max_params = max(100000, self.max_params // 2)
                    del model
                    continue
                raise
            island = assign_island(cell_config, n_cells, init_channels)

            candidates.append({
                'cell_config': cell_config,
                'n_cells': n_cells,
                'init_channels': init_channels,
                'naswot': naswot_score,
                'synflow': synflow_score,
                'params': param_count,
                'island': island,
                'descriptor': architecture_descriptor(
                    cell_config, n_cells, init_channels, param_count),
            })
            model.cpu()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"  Evaluated {len(candidates)} valid candidates in "
              f"{show_time(time.perf_counter() - search_start)}")

        if not candidates:
            print("  WARNING: No valid candidates found, using fallback model")
            return self._build_fallback_model()

        # ==================================================================
        # PHASE 2: Normalize scores and compute combined ranking
        # ==================================================================
        print("\n  Phase 2: Ranking candidates...")

        naswot_scores = robust_normalize([c['naswot'] for c in candidates])
        # log1p reduces SynFlow's strong width/depth scale bias before ranking.
        synflow_scores = robust_normalize([
            math.log1p(max(0.0, c['synflow'])) for c in candidates
        ])

        for i, c in enumerate(candidates):
            c['naswot_norm'] = naswot_scores[i]
            c['synflow_norm'] = synflow_scores[i]
            c['combined_score'] = compute_combined_score(
                naswot_scores[i], synflow_scores[i],
                naswot_weight=self.proxy_weights["naswot"]
            )

            # Param penalty only for simple tasks
            if self.is_simple_task:
                ideal = self.num_classes * 50_000
                penalty = max(0, (c['params'] - ideal * 3) / (ideal * 10))
                c['combined_score'] -= penalty

        candidates.sort(key=lambda c: c['combined_score'], reverse=True)

        print("\n  Phase 3: Measured diversity selection...")
        finalist_count = min(5 if self._time_left() > 3600 else 3, len(candidates))
        champions = select_diverse_candidates(candidates, finalist_count)
        diversity = diversity_summary(champions)
        self.metadata["diversity"] = diversity
        print("  Selected {} finalists | descriptor distance "
              "min={minimum:.3f}, mean={mean:.3f}, max={maximum:.3f}".format(
                  len(champions), **diversity))

        # Proxy scores filter the space. Equal-budget short training decides the
        # champion whenever the live budget can support it.
        best = self._successive_halving(champions)

        # Save runner-up configs in metadata so Trainer can try them if time permits
        runner_ups = []
        ranked_champions = sorted(
            champions,
            key=lambda c: c.get("short_accuracy", -1.0),
            reverse=True)
        for champ in ranked_champions:
            if champ is best:
                continue
            runner_ups.append({
                'cell_config': champ['cell_config'],
                'n_cells': champ['n_cells'],
                'init_channels': champ['init_channels'],
                'params': champ['params'],
                'island': champ['island'],
            })
            if len(runner_ups) == 2:
                break
        self.metadata['runner_up_configs'] = runner_ups
        self.metadata['dropout_rate'] = self.dropout_rate

        print(f"\n  Selected architecture:")
        print(f"    Island: {best['island']}")
        print(f"    Cells: {best['n_cells']}, Init channels: {best['init_channels']}")
        print(f"    Params: {best['params']:,}")
        print(f"    NASWOT (norm): {best['naswot_norm']:.4f}")
        print(f"    SynFlow (norm): {best['synflow_norm']:.4f}")
        print(f"    Combined score: {best['combined_score']:.4f}")
        if runner_ups:
            print(f"    Runner-ups saved: {len(runner_ups)} "
                  f"({', '.join(r['island'] for r in runner_ups)})")

        # Build the final model
        model = build_model_from_config(
            best['cell_config'], self.in_channels, self.num_classes,
            best['n_cells'], best['init_channels'], self.dropout_rate
        )

        # ==================================================================
        # PHASE 5: Short validation warm-up if time permits
        # ==================================================================
        elapsed = time.perf_counter() - search_start
        remaining_for_warmup = (self.time_remaining * 0.05)

        if remaining_for_warmup > 30 and elapsed < (self.time_remaining * 0.10):
            print(f"\n  Phase 5: Validation warm-up ({show_time(remaining_for_warmup)} budget)...")
            model = self._validation_warmup(model, max_time=min(remaining_for_warmup, 60))
        else:
            print(f"\n  Skipping validation warm-up (time constraint)")

        total_time = time.perf_counter() - search_start
        print(f"\n  NAS complete in {show_time(total_time)}")

        return model

    def _prediction_reserve(self):
        test_size = int(self.metadata.get("test_size", 0))
        baseline = max(45.0, 0.08 * max(0.0, self.time_remaining))
        return min(600.0, baseline + test_size * 0.002)

    def _successive_halving(self, finalists):
        """Short-train diverse finalists with identical update budgets."""
        if len(finalists) <= 1 or self._time_left() < 600:
            print("  Short finalist training skipped; using proxy leader")
            return max(finalists, key=lambda c: c["combined_score"])
        available = self._time_left() - self._prediction_reserve()
        phase_budget = min(300.0, max(60.0, available * 0.10))
        per_model = phase_budget / len(finalists)
        survivors = []
        for candidate in finalists:
            if self._time_left() <= self._prediction_reserve() + 90:
                break
            model = build_model_from_config(
                candidate["cell_config"], self.in_channels, self.num_classes,
                candidate["n_cells"], candidate["init_channels"],
                self.dropout_rate)
            accuracy = self._short_train_accuracy(model, per_model)
            candidate["short_accuracy"] = accuracy
            if accuracy >= 0.0:
                survivors.append(candidate)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("    {:<14} proxy={:.3f} short-val={}".format(
                candidate["island"], candidate["combined_score"],
                "{:.2%}".format(accuracy) if accuracy >= 0 else "failed"))
        if not survivors:
            return max(finalists, key=lambda c: c["combined_score"])
        survivors.sort(
            key=lambda c: (c["short_accuracy"], c["combined_score"]),
            reverse=True)
        return survivors[0]

    def _short_train_accuracy(self, model, seconds):
        model.to(self.device)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        started = time.perf_counter()
        steps = 0
        try:
            model.train()
            while time.perf_counter() - started < seconds * 0.70 and steps < 40:
                for data, target in self.train_loader:
                    if time.perf_counter() - started >= seconds * 0.70:
                        break
                    data = data.to(self.device)
                    target = target.to(self.device).long()
                    optimizer.zero_grad()
                    loss = criterion(model(data), target)
                    loss.backward()
                    optimizer.step()
                    steps += 1
            model.eval()
            correct = seen = 0
            with torch.no_grad():
                for data, target in self.valid_loader:
                    if time.perf_counter() - started >= seconds:
                        break
                    data = data.to(self.device)
                    target = target.to(self.device).long()
                    correct += (model(data).argmax(1) == target).sum().item()
                    seen += target.numel()
                    if seen >= 1024:
                        break
            return correct / float(seen) if seen else -1.0
        except RuntimeError as error:
            if "out of memory" in str(error).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
                return -1.0
            raise

    def _validation_warmup(self, model, max_time=60):
        """Short training warm-up to verify gradient flow is healthy."""
        model.to(self.device)
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        start = time.time()
        steps = 0
        max_steps = 10

        try:
            for data, target in self.train_loader:
                if time.time() - start > max_time or steps >= max_steps:
                    break
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                steps += 1
        except Exception as e:
            print(f"    Warm-up failed: {e}")

        print(f"    Warm-up: {steps} steps in {show_time(time.time() - start)}")

        model.cpu()
        return model

    def _build_fallback_model(self):
        """Fallback: build a simple default model if search fails."""
        cell_config = [
            ('conv3x3', 0),
            ('skip', 0),
            ('conv3x3', 1),
        ]
        model = build_model_from_config(
            cell_config,
            self.in_channels,
            self.num_classes,
            n_cells=3,
            init_channels=32,
            dropout_rate=self.dropout_rate,
        )
        return model
