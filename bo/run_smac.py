#!/usr/bin/env python3
"""Run 100 low-budget pipeline evaluations with SMAC.

The target function evaluates validation accuracy directly; test labels and the
competition scoring script are deliberately not used during optimization.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Float,
)
from smac import HyperparameterOptimizationFacade, Scenario
from smac.initial_design.sobol_design import SobolInitialDesign
from smac.main.config_selector import ConfigSelector


class BudgetClock:
    def __init__(self, seconds: float):
        self.deadline = time.perf_counter() + seconds

    def check(self) -> float:
        return self.deadline - time.perf_counter()


class DiverseConfigSelector(ConfigSelector):
    """Reject near-duplicates among recently issued parallel configurations."""

    _NUMERIC_RANGES = {
        "learning_rate_multiplier": (0.5, 2.0, True),
        "weight_decay_multiplier": (0.25, 4.0, True),
    }

    def __init__(
        self,
        scenario: Scenario,
        *,
        initial_design_size: int,
        diversity_threshold: float,
        diversity_window: int,
    ):
        super().__init__(scenario, retrain_after=1, max_new_config_tries=64)
        self._initial_design_size = initial_design_size
        self._diversity_threshold = diversity_threshold
        self._diversity_window = diversity_window
        self._issued = []

    @classmethod
    def _distance(cls, left, right) -> float:
        left_values = dict(left)
        right_values = dict(right)
        distances = []
        for name in left_values:
            a, b = left_values[name], right_values[name]
            if name in cls._NUMERIC_RANGES:
                lower, upper, logarithmic = cls._NUMERIC_RANGES[name]
                if logarithmic:
                    a, b = math.log(float(a)), math.log(float(b))
                    lower, upper = math.log(lower), math.log(upper)
                distances.append(
                    abs(float(a) - float(b)) / max(1e-12, upper - lower))
            else:
                distances.append(0.0 if a == b else 1.0)
        return sum(distances) / max(1, len(distances))

    def __iter__(self):
        rejected = 0
        for config in super().__iter__():
            recent = self._issued[-self._diversity_window:]
            initial = len(self._issued) < self._initial_design_size
            diverse = (
                not recent
                or min(self._distance(config, other) for other in recent)
                >= self._diversity_threshold
            )
            # Never interfere with the Sobol design. The retry escape prevents
            # a restrictive threshold from exhausting a small search space.
            if initial or diverse or rejected >= 64:
                self._issued.append(config)
                rejected = 0
                yield config
            else:
                rejected += 1


def configuration_space(seed: int) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=seed)
    learning_rate_multiplier = Float(
        "learning_rate_multiplier",
        bounds=(0.5, 2.0),
        default=1.0,
        log=True,
    )
    weight_decay_multiplier = Float(
        "weight_decay_multiplier",
        bounds=(0.25, 4.0),
        default=1.0,
        log=True,
    )
    label_smoothing = Categorical(
        "label_smoothing", [0.0, 0.025, 0.05, 0.1], default=0.1)
    batch_size = Categorical("batch_size", [16, 32, 64], default=64)
    cs.add([
        learning_rate_multiplier,
        weight_decay_multiplier,
        label_smoothing,
        batch_size,
    ])
    return cs


def effective_training_config(config: dict, metadata: dict) -> dict:
    """Resolve transferable multipliers against the incumbent's adaptations."""
    batch_size = int(config["batch_size"])
    base_lr = min(0.2, max(0.01, 0.05 * batch_size / 128.0))

    input_shape = metadata.get("input_shape", [50_000])
    num_classes = int(metadata.get("num_classes", 10))
    spatial_size = 1
    if len(input_shape) >= 4:
        spatial_size = int(input_shape[2]) * int(input_shape[3])
    base_weight_decay = (
        2e-3 if num_classes <= 10 and spatial_size <= 512 else 5e-4
    )

    return {
        "learning_rate": (
            base_lr * float(config["learning_rate_multiplier"])
        ),
        "weight_decay": (
            base_weight_decay * float(config["weight_decay_multiplier"])
        ),
        "label_smoothing": float(config["label_smoothing"]),
        "batch_size": batch_size,
    }


def _load_submission(submission_dir: Path):
    submission = str(submission_dir.resolve())
    if submission not in sys.path:
        sys.path.insert(0, submission)
    modules = [
        importlib.import_module(name)
        for name in ("data_processor", "nas", "trainer")
    ]
    return modules[0].DataProcessor, modules[1].NAS, modules[2].Trainer


def _load_dataset(dataset_dir: Path, train_limit: int | None):
    mmap_mode = "r" if train_limit is None else None
    train_x = np.load(dataset_dir / "train_x.npy", mmap_mode=mmap_mode)
    train_y = np.load(dataset_dir / "train_y.npy", mmap_mode=mmap_mode)
    valid_x = np.load(dataset_dir / "valid_x.npy", mmap_mode=mmap_mode)
    valid_y = np.load(dataset_dir / "valid_y.npy", mmap_mode=mmap_mode)
    test_x_path = dataset_dir / "test_x.npy"
    if train_limit is not None:
        train_x = train_x[:train_limit]
        train_y = train_y[:train_limit]
    # DataProcessor requires test data, but optimization never predicts on it.
    test_x = np.load(test_x_path, mmap_mode="r")[:1]
    with (dataset_dir / "metadata").open() as handle:
        metadata = json.load(handle)
    metadata["input_shape"] = list(train_x.shape)
    return train_x, train_y, valid_x, valid_y, test_x, metadata


class PipelineTarget:
    """Pickle-friendly SMAC target callable."""

    def __init__(
        self,
        submission_dir: str,
        dataset_dirs: list[str],
        budget_seconds: float,
        train_limit: int | None,
        objective: str,
        failure_cost: float,
    ):
        self.submission_dir = submission_dir
        self.dataset_dirs = dataset_dirs
        self.budget_seconds = budget_seconds
        self.train_limit = train_limit
        self.objective = objective
        self.failure_cost = failure_cost

    def __call__(self, config, seed: int = 0) -> float:
        started = time.perf_counter()
        config_dict = dict(config)
        accuracies: list[float] = []
        adjusted_scores: list[float] = []
        effective_configs: list[dict] = []
        DataProcessor, NAS, Trainer = _load_submission(
            Path(self.submission_dir))

        for dataset_name in self.dataset_dirs:
            dataset_dir = Path(dataset_name)
            try:
                trial_seed = int(seed)
                random.seed(trial_seed)
                np.random.seed(trial_seed % (2**32 - 1))
                torch.manual_seed(trial_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(trial_seed)

                data = _load_dataset(dataset_dir, self.train_limit)
                train_x, train_y, valid_x, valid_y, test_x, metadata = data
                clock = BudgetClock(self.budget_seconds)
                metadata["time_remaining"] = self.budget_seconds
                effective_config = effective_training_config(
                    config_dict, metadata)
                metadata["bo_config"] = effective_config
                metadata["seed"] = trial_seed
                metadata["disable_checkpoint"] = True
                effective_configs.append({
                    "dataset": dataset_dir.name,
                    **effective_config,
                })

                processor = DataProcessor(
                    train_x, train_y, valid_x, valid_y, test_x, metadata, clock)
                train_loader, valid_loader, _ = processor.process()
                model = NAS(train_loader, valid_loader, metadata, clock).search()
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
                trainer = Trainer(
                    model, device, train_loader, valid_loader, metadata, clock)
                trainer.train()
                accuracy = float(trainer._best_accuracy)
                if not math.isfinite(accuracy) or accuracy < 0:
                    raise RuntimeError("no validation accuracy was produced")
                accuracies.append(accuracy)
                benchmark = float(metadata.get("benchmark", 0.0))
                denominator = max(1e-12, 100.0 - benchmark)
                adjusted = (100.0 * accuracy - benchmark) * 10.0 / denominator
                adjusted_scores.append(max(-10.0, adjusted))
            except Exception:
                print(
                    "BO evaluation failed for {}:\n{}".format(
                        dataset_dir.name, traceback.format_exc()),
                    file=sys.stderr,
                )
                return self.failure_cost
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not accuracies:
            return self.failure_cost
        # SMAC minimizes. Equal dataset weighting prevents large datasets from
        # dominating the unseen-data robustness objective. The adjusted-score
        # option mirrors evaluation/score.py using validation accuracy.
        if self.objective == "adjusted_score":
            cost = -float(np.mean(adjusted_scores))
        else:
            cost = 1.0 - float(np.mean(accuracies))
        print(json.dumps({
            "bo_config": config_dict,
            "effective_training_configs": effective_configs,
            "accuracies": accuracies,
            "adjusted_scores": adjusted_scores,
            "cost": cost,
            "runtime_seconds": time.perf_counter() - started,
        }, sort_keys=True))
        return cost


def _dataset_dirs(root: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        paths = [
            root / name if not Path(name).is_absolute() else Path(name)
            for name in requested
        ]
    else:
        paths = sorted(path for path in root.iterdir() if path.is_dir())
    missing = [
        str(path) for path in paths
        if not (path / "train_x.npy").is_file()
        or not (path / "valid_y.npy").is_file()
        or not (path / "metadata").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "invalid dataset directories: {}".format(", ".join(missing)))
    if not paths:
        raise FileNotFoundError("no datasets found below {}".format(root))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument(
        "--datasets", nargs="+",
        help="Dataset directory names; defaults to every directory in the root.")
    parser.add_argument("--submission", default="submission_template")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--initial-sobol", type=int, default=25)
    parser.add_argument("--workers", type=int, default=25)
    parser.add_argument(
        "--budget-minutes", type=float, default=15.0,
        help="Wall-clock budget per dataset and configuration.")
    parser.add_argument(
        "--train-limit", type=int,
        help="Optional deterministic prefix of training examples for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="bo/output")
    parser.add_argument(
        "--diversity-threshold", type=float, default=0.08,
        help="Minimum normalized mixed-space distance to 25 recent proposals.")
    parser.add_argument(
        "--objective",
        choices=["adjusted_score", "mean_accuracy"],
        default="adjusted_score",
        help="Adjusted score mirrors evaluation/score.py on validation labels.")
    parser.add_argument(
        "--scheduler-address",
        help="Existing Dask scheduler, e.g. tcp://host:8786.")
    parser.add_argument(
        "--luh-slurm", action="store_true",
        help="Create LUH Slurm workers dynamically with dask-jobqueue.")
    parser.add_argument("--slurm-partition", default="ai")
    parser.add_argument(
        "--slurm-gpu",
        default="a100",
        help="GPU type (for gpu:TYPE:1); pass an empty value for --gres=gpu:1.")
    parser.add_argument("--slurm-cpus", type=int, default=2)
    parser.add_argument("--slurm-memory", default="16GiB")
    parser.add_argument("--slurm-walltime", default="03:00:00")
    parser.add_argument(
        "--slurm-conda-env",
        help="Conda environment available on every worker.")
    parser.add_argument(
        "--slurm-venv",
        help="Absolute virtualenv path available on every worker.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 1 or not 0 <= args.initial_sobol <= args.trials:
        raise ValueError("require 0 <= initial-sobol <= trials")
    if not 0.0 <= args.diversity_threshold <= 1.0:
        raise ValueError("diversity-threshold must be between zero and one")
    if args.scheduler_address and args.luh_slurm:
        raise ValueError(
            "--scheduler-address and --luh-slurm are mutually exclusive")
    if args.luh_slurm and bool(args.slurm_conda_env) == bool(args.slurm_venv):
        raise ValueError(
            "with --luh-slurm, provide exactly one of "
            "--slurm-conda-env or --slurm-venv")
    root = Path(args.datasets_root).resolve()
    datasets = _dataset_dirs(root, args.datasets)
    cs = configuration_space(args.seed)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    client = None
    cluster = None
    if args.scheduler_address:
        from distributed import Client
        client = Client(args.scheduler_address)
    elif args.luh_slurm:
        from dask_jobqueue import SLURMCluster
        from distributed import Client

        logs = output / "slurm-logs"
        logs.mkdir(parents=True, exist_ok=True)
        if args.slurm_venv:
            environment_prologue = [
                "module load GCCcore/.13.2.0 Python/3.11.5 CUDA/11.8.0",
                "source {}/bin/activate".format(
                    Path(args.slurm_venv).resolve()),
            ]
        else:
            environment_prologue = [
                "module load Miniforge3",
                "conda activate {}".format(args.slurm_conda_env),
            ]
        gpu_directive = (
            "--gres=gpu:{}:1".format(args.slurm_gpu)
            if args.slurm_gpu else "--gres=gpu:1"
        )
        cluster = SLURMCluster(
            queue=args.slurm_partition,
            cores=args.slurm_cpus,
            processes=1,
            memory=args.slurm_memory,
            walltime=args.slurm_walltime,
            job_name="nas_smac_worker",
            job_extra_directives=[
                "--nodes=1",
                "--ntasks=1",
                gpu_directive,
                "--output={}/%x_%j.out".format(logs),
                "--error={}/%x_%j.err".format(logs),
            ],
            job_script_prologue=environment_prologue + [
                "cd {}".format(Path.cwd().resolve()),
            ],
        )
        # dask-jobqueue translates this into at most `workers` Slurm jobs.
        cluster.scale(jobs=args.workers)
        client = Client(cluster)
        print("Dask dashboard: {}".format(client.dashboard_link))

    scenario = Scenario(
        cs,
        deterministic=False,
        n_trials=args.trials,
        n_workers=args.workers,
        output_directory=output,
        seed=args.seed,
    )
    initial_design = SobolInitialDesign(
        scenario=scenario,
        n_configs=args.initial_sobol,
        max_ratio=1.0,
    )
    config_selector = DiverseConfigSelector(
        scenario,
        initial_design_size=args.initial_sobol,
        diversity_threshold=args.diversity_threshold,
        diversity_window=args.workers,
    )
    intensifier = HyperparameterOptimizationFacade.get_intensifier(
        scenario, max_config_calls=1)
    target = PipelineTarget(
        str(Path(args.submission).resolve()),
        [str(path) for path in datasets],
        args.budget_minutes * 60.0,
        args.train_limit,
        args.objective,
        failure_cost=10.0 if args.objective == "adjusted_score" else 1.0,
    )
    smac = HyperparameterOptimizationFacade(
        scenario,
        target,
        initial_design=initial_design,
        config_selector=config_selector,
        intensifier=intensifier,
        overwrite=False,
        dask_client=client,
    )
    try:
        incumbent = smac.optimize()
        result = {
            "incumbent": dict(incumbent),
            "datasets": [path.name for path in datasets],
            "trials": args.trials,
            "initial_sobol": args.initial_sobol,
            "workers": args.workers,
            "budget_minutes_per_dataset": args.budget_minutes,
            "objective": args.objective,
            "diversity_threshold": args.diversity_threshold,
        }
        with (output / "incumbent.json").open("w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()


if __name__ == "__main__":
    main()
