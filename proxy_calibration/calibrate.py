#!/usr/bin/env python3
"""Calibrate NASWOT/SynFlow against equal-budget short-trained accuracy.

This is an offline cluster experiment. It imports the exact competition search
space but never modifies datasets or invokes the official evaluation harness.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMISSION = os.path.join(ROOT, "submission_template")
sys.path.insert(0, SUBMISSION)

from data_processor import DataProcessor  # noqa: E402
from helpers import (  # noqa: E402
    architecture_descriptor,
    build_model_from_config,
    compute_naswot_score,
    compute_param_count,
    compute_synflow_score,
    robust_normalize,
    sample_cell_config,
)


class CalibrationClock:
    def __init__(self, seconds):
        self.deadline = time.perf_counter() + seconds

    def check(self):
        return self.deadline - time.perf_counter()


def rankdata(values):
    """Average ranks for ties, implemented without scipy."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average = 0.5 * (position + end - 1)
        for offset in range(position, end):
            ranks[order[offset]] = average
        position = end
    return ranks


def correlation(left, right):
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.std() == 0 or right.std() == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def spearman(left, right):
    return correlation(rankdata(left), rankdata(right))


def kendall(left, right):
    concordant = discordant = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            product = (left[i] - left[j]) * (right[i] - right[j])
            concordant += product > 0
            discordant += product < 0
    total = concordant + discordant
    return float(concordant - discordant) / total if total else 0.0


def load_dataset(path):
    with open(os.path.join(path, "metadata"), "r") as handle:
        metadata = json.load(handle)
    arrays = [
        np.load(os.path.join(path, name), mmap_mode="r")
        for name in ("train_x.npy", "train_y.npy", "valid_x.npy",
                     "valid_y.npy", "test_x.npy")
    ]
    return arrays, metadata


def short_train(model, train_loader, valid_loader, device, updates):
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    iterator = iter(train_loader)
    model.train()
    completed = 0
    while completed < updates:
        try:
            data, target = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            data, target = next(iterator)
        data = data.to(device)
        target = target.to(device).long()
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        completed += 1
    model.eval()
    correct = seen = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data = data.to(device)
            target = target.to(device).long()
            correct += (model(data).argmax(1) == target).sum().item()
            seen += target.numel()
    return correct / float(max(1, seen))


def candidate_ranges(metadata, diagnostics):
    if diagnostics.get("sequence_grid_likely", False):
        return 3, [2, 3, 4], [16, 24, 32]
    spatial = diagnostics["spatial_size"]
    classes = int(metadata["num_classes"])
    if classes <= 10 and spatial <= 512:
        return 3, [2, 3], [16, 24]
    if classes <= 50:
        return 4, [3, 4, 5], [24, 32, 48]
    return 5, [3, 4, 5], [24, 32, 48]


def calibrate_dataset(path, args, device):
    arrays, metadata = load_dataset(path)
    train_x, train_y, valid_x, valid_y, test_x = arrays
    metadata["time_remaining"] = args.timeout_minutes * 60.0
    processor = DataProcessor(
        train_x, train_y, valid_x, valid_y, test_x, metadata,
        CalibrationClock(args.timeout_minutes * 60.0))
    train_loader, valid_loader, _ = processor.process()
    nodes, cell_counts, channels = candidate_ranges(
        metadata, metadata["diagnostics"])
    position_sensitive = bool(
        metadata["diagnostics"].get("sequence_grid_likely", False))
    rng = random.Random(args.seed)
    records = []
    for index in range(args.candidates):
        config = sample_cell_config(nodes, rng)
        n_cells = rng.choice(cell_counts)
        init_channels = rng.choice(channels)
        torch.manual_seed(args.seed + index)
        model = build_model_from_config(
            config, processor.train_x.shape[1], metadata["num_classes"],
            n_cells, init_channels,
            dropout_rate=0.15 if position_sensitive else 0.1,
            input_height=processor.train_x.shape[2],
            input_width=processor.train_x.shape[3],
            position_sensitive=position_sensitive)
        params = compute_param_count(model)
        try:
            naswot = compute_naswot_score(
                model, train_loader, device, max_samples=args.proxy_batch)
            synflow = compute_synflow_score(model, train_loader, device)
            accuracy = short_train(
                model, train_loader, valid_loader, device, args.updates)
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            naswot = synflow = accuracy = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        record = {
            "index": index,
            "cell_config": config,
            "n_cells": n_cells,
            "init_channels": init_channels,
            "params": params,
            "descriptor": architecture_descriptor(
                config, n_cells, init_channels, params),
            "naswot": naswot,
            "synflow": synflow,
            "short_accuracy": accuracy,
        }
        records.append(record)
        print("{} [{:02d}/{:02d}] params={} val={}".format(
            metadata.get("codename", os.path.basename(path)), index + 1,
            args.candidates, params,
            "OOM" if accuracy is None else "{:.3%}".format(accuracy)))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "dataset": metadata.get("codename", os.path.basename(path)),
        "records": records,
    }


def summarize(results):
    summaries = {}
    pooled = []
    for result in results:
        records = [r for r in result["records"]
                   if r["short_accuracy"] is not None]
        pooled.extend(records)
        accuracy = [r["short_accuracy"] for r in records]
        summaries[result["dataset"]] = {
            "n": len(records),
            "naswot_spearman": spearman(
                [r["naswot"] for r in records], accuracy),
            "synflow_spearman": spearman(
                [math.log1p(max(0.0, r["synflow"])) for r in records],
                accuracy),
            "params_spearman": spearman(
                [r["params"] for r in records], accuracy),
            "naswot_kendall": kendall(
                [r["naswot"] for r in records], accuracy),
            "synflow_kendall": kendall(
                [r["synflow"] for r in records], accuracy),
        }
    def score_weight(result_subset, weight):
        dataset_scores = []
        for result in result_subset:
            records = [r for r in result["records"]
                       if r["short_accuracy"] is not None]
            if len(records) < 2:
                continue
            nw = robust_normalize([r["naswot"] for r in records])
            sf = robust_normalize([
                math.log1p(max(0.0, r["synflow"])) for r in records])
            combined = [weight * a + (1.0 - weight) * b
                        for a, b in zip(nw, sf)]
            dataset_scores.append(spearman(
                combined, [r["short_accuracy"] for r in records]))
        return sum(dataset_scores) / max(1, len(dataset_scores))

    best_weight = 0.5
    best_score = -float("inf")
    for weight_percent in range(0, 101, 5):
        weight = weight_percent / 100.0
        score = score_weight(results, weight)
        if score > best_score:
            best_score, best_weight = score, weight
    leave_one_out = []
    if len(results) > 1:
        for held_out in results:
            training = [result for result in results if result is not held_out]
            selected = max(
                (percent / 100.0 for percent in range(0, 101, 5)),
                key=lambda weight: score_weight(training, weight))
            leave_one_out.append({
                "held_out": held_out["dataset"],
                "selected_naswot_weight": selected,
                "held_out_spearman": score_weight([held_out], selected),
            })
    return summaries, {
        "naswot_weight": best_weight,
        "synflow_weight": 1.0 - best_weight,
        "mean_dataset_spearman": best_score,
        "leave_one_dataset_out": leave_one_out,
        "note": "Copy this file to submission_template/proxy_weights.json",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Dataset directories containing metadata and npy files")
    parser.add_argument("--output", default="proxy_calibration/results.json")
    parser.add_argument("--candidates", type=int, default=30)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--proxy-batch", type=int, default=24)
    parser.add_argument("--timeout-minutes", type=float, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Calibration device: {}".format(device))
    results = [calibrate_dataset(path, args, device) for path in args.datasets]
    summaries, weights = summarize(results)
    payload = {"summaries": summaries, "recommended_weights": weights,
               "datasets": results}
    output = os.path.abspath(args.output)
    directory = os.path.dirname(output)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(output, "w") as handle:
        json.dump(payload, handle, indent=2)
    weights_path = os.path.join(directory, "proxy_weights.json")
    with open(weights_path, "w") as handle:
        json.dump(weights, handle, indent=2)
    print(json.dumps({"summaries": summaries, "weights": weights}, indent=2))
    print("Wrote {} and {}".format(output, weights_path))


if __name__ == "__main__":
    main()
