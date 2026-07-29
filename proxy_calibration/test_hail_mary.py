"""Deterministic smoke tests for robust search and preprocessing."""

import pathlib
import sys
import unittest

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission_template"))

from data_processor import (  # noqa: E402
    DataProcessor,
    _RUN_BUDGET_STATE,
    allocate_dataset_budget,
    dataset_diagnostics,
)
from helpers import (  # noqa: E402
    architecture_descriptor,
    build_model_from_config,
    descriptor_distance,
    diversity_summary,
    robust_normalize,
    select_diverse_candidates,
)
from trainer import Trainer  # noqa: E402


class Clock:
    def check(self):
        return 300.0


class HailMaryTests(unittest.TestCase):
    def test_global_final_budget_is_shared(self):
        original = dict(_RUN_BUDGET_STATE)
        try:
            _RUN_BUDGET_STATE.update({
                "datasets_seen": 0,
                "global_24h_mode": False,
            })
            metadata = {"time_remaining": 24 * 3600}
            allocated = allocate_dataset_budget(metadata, Clock())
            self.assertEqual(metadata["budget_mode"], "global-share")
            self.assertAlmostEqual(
                allocated, 0.97 * 24 * 3600 / 3)
        finally:
            _RUN_BUDGET_STATE.update(original)

    def test_rank_normalization_handles_nonfinite(self):
        result = robust_normalize([1.0, float("nan"), 3.0])
        self.assertEqual(result[1], 0.0)
        self.assertEqual(result[2], 1.0)

    def test_diverse_selection(self):
        configs = [
            ([("skip", 0)] * 3, 2, 16),
            ([("conv3x3", 0)] * 3, 3, 32),
            ([("max_pool", 0)] * 3, 5, 64),
        ]
        candidates = [{
            "combined_score": 1.0 - 0.1 * index,
            "descriptor": architecture_descriptor(
                config, cells, channels, 10000 * (index + 1)),
        } for index, (config, cells, channels) in enumerate(configs)]
        selected = select_diverse_candidates(candidates, 3)
        self.assertEqual(len(selected), 3)
        self.assertGreater(diversity_summary(selected)["minimum"], 0.0)
        self.assertGreater(descriptor_distance(
            selected[0]["descriptor"], selected[1]["descriptor"]), 0.0)

    def test_encoded_data_and_label_remapping(self):
        x = np.zeros((9, 8, 8), dtype=np.uint8)
        x[:, 2:4, 2:4] = 1
        y = np.asarray([10, 20, 30] * 3)
        metadata = {
            "num_classes": 3, "codename": "opaque",
            "input_shape": [9, 1, 8, 8], "time_remaining": 300.0,
        }
        processor = DataProcessor(
            x, y, x[:3], y[:3], x[:4], metadata, Clock())
        train, _, test = processor.process()
        self.assertTrue(metadata["diagnostics"]["encoded_likely"])
        self.assertEqual(metadata["augmentation_policy"], [])
        self.assertEqual(metadata["label_values"], [10, 20, 30])
        self.assertEqual(len(train.dataset), 9)
        self.assertEqual(len(test.dataset), 4)

    def test_nonfinite_diagnostics(self):
        x = np.ones((4, 1, 4, 4), dtype=np.float32)
        x[0, 0, 0, 0] = np.nan
        diagnostics = dataset_diagnostics(x, [0, 1, 0, 1], 2)
        self.assertGreater(diagnostics["nonfinite_fraction"], 0.0)

    def test_positional_specialist_is_optional_and_bounded(self):
        rng = np.random.RandomState(42)
        x = np.zeros((16, 1, 24, 24), dtype=np.float32)
        columns = np.arange(24)
        for index in range(len(x)):
            x[index, 0, rng.randint(0, 24, size=24), columns] = 1.0
        diagnostics = dataset_diagnostics(
            x, np.arange(len(x)) % 10, 10)
        self.assertTrue(diagnostics["sequence_grid_likely"])

        config = [
            ("conv3x3", 0),
            ("sep3x3", 0),
            ("skip", 1),
        ]
        baseline = build_model_from_config(
            config, 1, 10, 4, 32, 0.1)
        specialist = build_model_from_config(
            config, 1, 10, 4, 32, 0.15,
            max_downsamples=1, spatial_pool_size=3)
        self.assertGreater(
            sum(layer is not None for layer in baseline.downsamples), 1)
        self.assertEqual(
            sum(layer is not None for layer in specialist.downsamples), 1)
        self.assertEqual(specialist.classifier.in_features, 64 * 3 * 3)
        self.assertEqual(
            specialist(torch.zeros(2, 1, 24, 24)).shape, (2, 10))

    def test_successive_halving_preserves_specialist_probe(self):
        states = [
            {
                "best_accuracy": score,
                "failed": False,
                "config": {"specialist": specialist},
            }
            for score, specialist in [
                (0.9, False), (0.8, False), (0.2, True)
            ]
        ]
        promoted = Trainer._promote(states, 1)
        self.assertEqual(len(promoted), 2)
        self.assertTrue(any(
            state["config"]["specialist"] for state in promoted))

    def test_validation_halves_are_stratified(self):
        targets = torch.tensor([0, 0, 0, 1, 1, 1])
        first, second = Trainer._stratified_half_masks(targets)
        self.assertTrue(torch.all(first ^ second))
        self.assertTrue(bool(first.any()))
        self.assertTrue(bool(second.any()))


if __name__ == "__main__":
    unittest.main()
