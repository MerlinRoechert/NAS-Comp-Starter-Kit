"""
nas.py - Neural Architecture Search using training-free proxies (NASWOT + SynFlow),
a cell-based search space, diversity islands, and budget-aware candidate selection.

Key design: Proxies are used as a FILTER (Phase 1-3) to cheaply narrow down candidates.
Final champion selection (Phase 4) uses actual short training to verify which
architecture truly learns best — addressing proxy unreliability.
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
        # (Proxies used as cheap FILTER, not final decision)
        # ==================================================================
        candidates = []
        print(f"\n  Phase 1: Sampling and scoring {self.n_candidates} candidates...")

        for i in range(self.n_candidates):
            elapsed = time.time() - search_start
            budget_for_search = self.time_remaining * 0.10  # 10% for proxy scoring
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
        # PHASE 4: Train island champions briefly — select by REAL validation
        # accuracy, not proxy score. This addresses proxy unreliability.
        # ==================================================================
        elapsed = time.time() - search_start
        # Allocate up to 8% of total budget for champion comparison training
        champion_budget = self.time_remaining * 0.08
        time_spent = time.time() - search_start
        remaining_for_champions = champion_budget - time_spent * 0.1

        if len(champions) > 1 and remaining_for_champions > 30:
            print(f"\n  Phase 4: Training {len(champions)} champions to select best "
                  f"(budget: {show_time(remaining_for_champions)})...")

            # Time per champion: split budget evenly
            time_per_champion = remaining_for_champions / len(champions)
            best_val_acc = -1.0
            best_champion = champions[0]

            for idx, champ in enumerate(champions):
                # Check time
                if time.time() - search_start > self.time_remaining * 0.18:
                    print(f"    Skipping remaining champions (time constraint)")
                    break

                model = build_model_from_config(
                    champ['cell_config'], self.in_channels, self.num_classes,
                    champ['n_cells'], champ['init_channels'], self.dropout_rate
                )
                val_acc = self._short_train_eval(model, max_time=time_per_champion)
                champ['val_acc'] = val_acc
                print(f"    {champ['island']} ({champ['params']:,} params): "
                      f"val_acc={val_acc:.2f}%")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_champion = champ

            best = best_champion
            print(f"\n  Winner: {best['island']} with val_acc={best_val_acc:.2f}%")
        else:
            # Not enough time for champion training — fall back to proxy ranking
            best = champions[0]
            print(f"\n  Phase 4: Skipped (using proxy ranking)")

        print(f"\n  Selected architecture:")
        print(f"    Island: {best['island']}")
        print(f"    Cells: {best['n_cells']}, Init channels: {best['init_channels']}")
        print(f"    Params: {best['params']:,}")
        print(f"    Combined proxy score: {best['combined_score']:.4f}")

        # Build the final model (fresh weights — trainer will do full training)
        model = build_model_from_config(
            best['cell_config'], self.in_channels, self.num_classes,
            best['n_cells'], best['init_channels'], self.dropout_rate
        )

        total_time = time.time() - search_start
        print(f"\n  NAS complete in {show_time(total_time)}")

        return model

    def _short_train_eval(self, model, max_time=60, max_epochs=5):
        """
        Train a model briefly and return validation accuracy.
        Used to compare island champions with real training signal
        instead of relying solely on proxy scores.
        """
        model.to(self.device)
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9,
                                    weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        start = time.time()

        for epoch in range(max_epochs):
            if time.time() - start > max_time:
                break
            for data, target in self.train_loader:
                if time.time() - start > max_time:
                    break
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.valid_loader:
                if time.time() - start > max_time + 30:  # small grace for eval
                    break
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                correct += (output.argmax(1) == target).sum().item()
                total += target.size(0)

        # Clean up GPU memory
        model.cpu()
        del optimizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return 100.0 * correct / max(1, total)

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
