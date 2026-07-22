"""
nas.py - Neural Architecture Search using training-free proxies (NASWOT + SynFlow),
a cell-based search space, diversity islands, and budget-aware candidate selection.
"""

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
    normalize_scores,
    compute_combined_score,
    assign_island,
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

    def _configure_search_space(self):
        """
        Configure search space parameters based on dataset metadata.
        Larger/more complex datasets get wider/deeper networks.
        Model size is scaled to dataset complexity to avoid overfitting.
        """
        n_datapoints = self.input_shape[0]
        spatial_size = self.img_height * self.img_width

        # Nodes per cell: 3-5 depending on complexity
        if self.num_classes <= 10 and spatial_size <= 1024:
            self.n_nodes = 3
        elif self.num_classes <= 50:
            self.n_nodes = 4
        else:
            self.n_nodes = 5

        # (B) Scale model size to dataset complexity
        # For simple tasks (few classes, tiny images), use smaller models
        if self.num_classes <= 10 and spatial_size <= 512:
            # Very simple tasks (e.g., Gutenberg: 6 classes, 27x18)
            self.cell_counts = [2, 3]
            self.init_channels_options = [16, 24]
        elif spatial_size <= 256:  # very small images (e.g., 16x16 or smaller)
            self.cell_counts = [2, 3, 4]
            self.init_channels_options = [16, 24, 32]
        elif spatial_size <= 1024:  # small images (e.g., 32x32)
            self.cell_counts = [3, 4, 5]
            self.init_channels_options = [24, 32, 48]
        else:  # larger images (e.g., 64x64)
            self.cell_counts = [4, 5, 6]
            self.init_channels_options = [32, 48, 64]

        # (A) Ideal param count heuristic — used for penalty in scoring
        self.ideal_params = self.num_classes * 50_000  # ~50K params per class

        # (C) Budget-aware: how many candidates to evaluate
        # Larger images need more exploration
        if self.time_remaining > 18000:  # > 5 hours
            self.n_candidates = 80
        elif self.time_remaining > 7200:  # > 2 hours
            self.n_candidates = 50
        elif self.time_remaining > 3600:  # > 1 hour
            base = 30
            # More candidates for larger images (need more exploration)
            if spatial_size > 1024:
                base = 40
            self.n_candidates = base
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
    4. Select the best champion from the best island
    5. Optionally do a short validation warm-up if time permits
    """
    def search(self):
        print(f"  NAS Search | Device: {self.device} | Candidates: {self.n_candidates}")
        print(f"  Dataset: {self.in_channels}ch, {self.img_height}x{self.img_width}, "
              f"{self.num_classes} classes")
        print(f"  Search space: {self.n_nodes} nodes/cell, "
              f"cells={self.cell_counts}, channels={self.init_channels_options}")

        search_start = time.time()
        rng = random.Random(42)

        # PHASE 1: Sample candidates and score with training-free proxies
        candidates = []
        print(f"\n  Phase 1: Sampling and scoring {self.n_candidates} candidates...")

        for i in range(self.n_candidates):
            # Check time budget (leave at least 80% for training)
            elapsed = time.time() - search_start
            budget_for_search = self.time_remaining * 0.15  # use at most 15% for NAS
            if elapsed > budget_for_search:
                print(f"  Time budget reached after {i} candidates ({show_time(elapsed)})")
                break

            # Sample architecture configuration
            cell_config = sample_cell_config(self.n_nodes, rng)
            n_cells = rng.choice(self.cell_counts)
            init_channels = rng.choice(self.init_channels_options)

            # Build model
            try:
                model = build_model_from_config(
                    cell_config, self.in_channels, self.num_classes,
                    n_cells, init_channels
                )
            except Exception:
                continue

            # Compute parameter count — skip if too large relative to task complexity
            param_count = compute_param_count(model)
            max_params = min(10_000_000, self.ideal_params * 20)  # adaptive cap
            if param_count > max_params:
                continue

            # Compute NASWOT score
            naswot_score = compute_naswot_score(model, self.train_loader, self.device)

            # Compute SynFlow score
            synflow_score = compute_synflow_score(model, self.train_loader, self.device)

            # Classify into island
            island = assign_island(cell_config, n_cells, init_channels)

            candidates.append({
                'cell_config': cell_config,
                'n_cells': n_cells,
                'init_channels': init_channels,
                'naswot': naswot_score,
                'synflow': synflow_score,
                'params': param_count,
                'island': island,
            })

        print(f"  Evaluated {len(candidates)} valid candidates in {show_time(time.time() - search_start)}")

        if not candidates:
            # Fallback: return a simple default model
            print("  WARNING: No valid candidates found, using fallback model")
            return self._build_fallback_model()

        # PHASE 2: Normalize scores and compute combined ranking
        print("\n  Phase 2: Ranking candidates...")

        naswot_scores = normalize_scores([c['naswot'] for c in candidates])
        synflow_scores = normalize_scores([c['synflow'] for c in candidates])

        for i, c in enumerate(candidates):
            c['naswot_norm'] = naswot_scores[i]
            c['synflow_norm'] = synflow_scores[i]
            c['combined_score'] = compute_combined_score(
                naswot_scores[i], synflow_scores[i], naswot_weight=0.5
            )

            # (A) Param count penalty — penalize overly large models for simple tasks
            param_penalty = max(0, (c['params'] - self.ideal_params * 3) / (self.ideal_params * 10))
            c['combined_score'] -= param_penalty

        # Sort by combined score (descending)
        candidates.sort(key=lambda c: c['combined_score'], reverse=True)

        # PHASE 3: Diversity islands — pick best from each island
        print("\n  Phase 3: Diversity island selection...")

        islands = {}
        for c in candidates:
            island_name = c['island']
            if island_name not in islands:
                islands[island_name] = []
            islands[island_name].append(c)

        print(f"  Islands: {', '.join(f'{k}({len(v)})' for k, v in islands.items())}")

        # Get champion from each island (best combined score)
        champions = []
        for island_name, members in islands.items():
            # Already sorted globally, so first member in each island is its champion
            champion = members[0]
            champions.append(champion)
            print(f"    {island_name} champion: score={champion['combined_score']:.4f}, "
                  f"params={champion['params']:,}")

        # PHASE 4: Select overall best architecture
        # Pick the overall champion (highest combined score)
        champions.sort(key=lambda c: c['combined_score'], reverse=True)
        best = champions[0]

        print(f"\n  Selected architecture:")
        print(f"    Island: {best['island']}")
        print(f"    Cells: {best['n_cells']}, Init channels: {best['init_channels']}")
        print(f"    Params: {best['params']:,}")
        print(f"    NASWOT (norm): {best['naswot_norm']:.4f}")
        print(f"    SynFlow (norm): {best['synflow_norm']:.4f}")
        print(f"    Combined score: {best['combined_score']:.4f}")

        # Build the final model
        model = build_model_from_config(
            best['cell_config'], self.in_channels, self.num_classes,
            best['n_cells'], best['init_channels']
        )

        # PHASE 5 (Optional): Short validation warm-up if time permits
        elapsed = time.time() - search_start
        remaining_for_warmup = (self.time_remaining * 0.05)  # 5% of total budget

        if remaining_for_warmup > 30 and elapsed < (self.time_remaining * 0.10):
            print(f"\n  Phase 5: Validation warm-up ({show_time(remaining_for_warmup)} budget)...")
            model = self._validation_warmup(model, max_time=min(remaining_for_warmup, 60))
        else:
            print(f"\n  Skipping validation warm-up (time constraint)")

        total_time = time.time() - search_start
        print(f"\n  NAS complete in {show_time(total_time)}")

        return model

    def _validation_warmup(self, model, max_time=60):
        """
        Short training warm-up on validation data to verify the model trains correctly.
        This is NOT full training — just a few steps to ensure gradient flow is healthy.
        """
        model.to(self.device)
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        start = time.time()
        steps = 0
        max_steps = 10  # very few steps

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

        # Move model back to CPU for trainer to handle device placement
        model.cpu()
        return model

    def _build_fallback_model(self):
        """
        Fallback: build a simple model if the search fails.
        Uses a conservative 3-cell architecture with conv3x3 operations.
        """
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
        )
        return model