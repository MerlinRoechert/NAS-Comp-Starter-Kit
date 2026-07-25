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
        Key insight: only constrain model size for truly simple tasks that are
        prone to overfitting. For complex tasks, allow large models.
        """
        n_datapoints = self.input_shape[0]
        spatial_size = self.img_height * self.img_width

        # Estimate dataset complexity
        self.is_simple_task = (self.num_classes <= 10 and spatial_size <= 512)
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
            self.max_params = 500_000
            self.dropout_rate = 0.3
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

        search_start = time.time()
        rng = random.Random(42)

        # ==================================================================
        # PHASE 1: Sample candidates and score with training-free proxies
        # ==================================================================
        candidates = []
        print(f"\n  Phase 1: Sampling and scoring {self.n_candidates} candidates...")

        for i in range(self.n_candidates):
            elapsed = time.time() - search_start
            budget_for_search = self.time_remaining * 0.12
            if elapsed > budget_for_search:
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

            naswot_score = compute_naswot_score(model, self.train_loader, self.device)
            synflow_score = compute_synflow_score(model, self.train_loader, self.device)
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

        print(f"  Evaluated {len(candidates)} valid candidates in "
              f"{show_time(time.time() - search_start)}")

        if not candidates:
            print("  WARNING: No valid candidates found, using fallback model")
            return self._build_fallback_model()

        # ==================================================================
        # PHASE 2: Normalize scores and compute combined ranking
        # ==================================================================
        print("\n  Phase 2: Ranking candidates...")

        naswot_scores = normalize_scores([c['naswot'] for c in candidates])
        synflow_scores = normalize_scores([c['synflow'] for c in candidates])

        for i, c in enumerate(candidates):
            c['naswot_norm'] = naswot_scores[i]
            c['synflow_norm'] = synflow_scores[i]
            c['combined_score'] = compute_combined_score(
                naswot_scores[i], synflow_scores[i], naswot_weight=0.5
            )

            # Param penalty only for simple tasks
            if self.is_simple_task:
                ideal = self.num_classes * 50_000
                penalty = max(0, (c['params'] - ideal * 3) / (ideal * 10))
                c['combined_score'] -= penalty

        candidates.sort(key=lambda c: c['combined_score'], reverse=True)

        # ==================================================================
        # PHASE 3: Diversity islands — pick best from each island
        # ==================================================================
        print("\n  Phase 3: Diversity island selection...")

        islands = {}
        for c in candidates:
            island_name = c['island']
            if island_name not in islands:
                islands[island_name] = []
            islands[island_name].append(c)

        print(f"  Islands: {', '.join(f'{k}({len(v)})' for k, v in islands.items())}")

        champions = []
        for island_name, members in islands.items():
            champion = members[0]
            champions.append(champion)
            print(f"    {island_name} champion: score={champion['combined_score']:.4f}, "
                  f"params={champion['params']:,}")

        # ==================================================================
        # PHASE 4: Select overall best architecture by proxy score
        # ==================================================================
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
            best['n_cells'], best['init_channels'], self.dropout_rate
        )

        # ==================================================================
        # PHASE 5: Short validation warm-up if time permits
        # ==================================================================
        elapsed = time.time() - search_start
        remaining_for_warmup = (self.time_remaining * 0.05)

        if remaining_for_warmup > 30 and elapsed < (self.time_remaining * 0.10):
            print(f"\n  Phase 5: Validation warm-up ({show_time(remaining_for_warmup)} budget)...")
            model = self._validation_warmup(model, max_time=min(remaining_for_warmup, 60))
        else:
            print(f"\n  Skipping validation warm-up (time constraint)")

        total_time = time.time() - search_start
        print(f"\n  NAS complete in {show_time(total_time)}")

        return model

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
