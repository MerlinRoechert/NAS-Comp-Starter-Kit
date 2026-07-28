"""Deterministic smoke tests for robust search and preprocessing."""

import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission_template"))

from data_processor import DataProcessor, dataset_diagnostics  # noqa: E402
from helpers import (  # noqa: E402
    architecture_descriptor,
    build_model_from_config,
    descriptor_distance,
    diversity_summary,
    robust_normalize,
    select_diverse_candidates,
)


class Clock:
    def check(self):
        return 300.0


class HailMaryTests(unittest.TestCase):
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

    def test_sequence_grid_preserves_position(self):
        rng = np.random.RandomState(42)
        x = np.zeros((16, 1, 24, 24), dtype=np.float32)
        columns = np.arange(24)
        for index in range(len(x)):
            x[index, 0, rng.randint(0, 24, size=24), columns] = 1.0
        diagnostics = dataset_diagnostics(
            x, np.arange(len(x)) % 10, 10)
        self.assertTrue(diagnostics["sequence_grid_likely"])
        self.assertAlmostEqual(
            diagnostics["column_one_hot_fraction"], 1.0)
        transposed = dataset_diagnostics(
            x.transpose(0, 1, 3, 2), np.arange(len(x)) % 10, 10)
        self.assertTrue(transposed["sequence_grid_likely"])
        self.assertAlmostEqual(
            transposed["row_one_hot_fraction"], 1.0)

        config = [
            ("conv3x3", 0),
            ("sep3x3", 0),
            ("skip", 1),
        ]
        model = build_model_from_config(
            config, 1, 10, 4, 32, 0.15,
            input_height=24, input_width=24,
            position_sensitive=True)
        self.assertEqual(
            sum(layer is not None for layer in model.downsamples), 1)
        self.assertEqual(model.classifier.in_features, 64 * 12 * 12)


if __name__ == "__main__":
    unittest.main()
