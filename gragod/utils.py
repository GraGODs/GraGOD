import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from colorama import Fore

from gragod.types import PathType


def get_logger(logger_name: str | None = None):
    if logger_name is None:
        logger_name = os.path.basename(__file__).split(".")[0]
    logger = logging.getLogger(logger_name)

    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(name)-5s %(levelname)-8s %(message)s")
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    return logger


def load_checkpoint_path(checkpoint_path: str, experiment_name: str) -> Path:
    """
    Load the checkpoint path from the checkpoint folder.
    If the checkpoint folder ends with ".ckpt", it is used as the checkpoint path.
    Otherwise, the checkpoint path is the best.ckpt file in the checkpoint folder.

    If the checkpoint path does not exist, it tries to load the checkpoint from the
    checkpoint folder with the experiment name.

    Args:
        checkpoint_path: The path to the checkpoint.
        experiment_name: The name of the experiment.

    Returns:
        The checkpoint path.
    """
    if not checkpoint_path.endswith(".ckpt"):
        checkpoint_path = os.path.join(checkpoint_path, "best.ckpt")

    if not os.path.exists(checkpoint_path):
        checkpoint_path_candidate = os.path.join(
            checkpoint_path,
            f"{experiment_name}.ckpt",
        )
        print(
            Fore.YELLOW + f"Checkpoint not found at {checkpoint_path}. "
            f"Trying with {checkpoint_path_candidate}" + Fore.RESET
        )

        if not os.path.exists(checkpoint_path_candidate):
            print(
                Fore.YELLOW
                + f"Tried with {checkpoint_path_candidate}, but it does not exist."
                + Fore.RESET
            )
            raise ValueError(
                f"Checkpoint not found at {checkpoint_path} or"
                f"{checkpoint_path_candidate}"
            )

    return Path(checkpoint_path)


def jit_compile_model(input_example: torch.Tensor, model, save_dir: PathType):
    with torch.jit.optimized_execution(True):
        traced_model = torch.jit.trace(model, input_example)

    print(f"Saving model in {save_dir}")
    torch.jit.save(traced_model, save_dir)


def pytest_is_running():
    return any(arg.startswith("pytest") for arg in sys.argv)


def set_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        print("No GPU or MPS available, training on CPU")
        device = "cpu"

    return device


def count_anomaly_ranges(labels: pd.DataFrame):
    """Count contiguous sequences of 1s in each time series column"""
    results = []

    # Convert tensor to numpy for easier manipulation
    labels_np = labels.to_numpy()
    if labels.ndim == 1:
        labels_np = labels_np.reshape(-1, 1)

    for col in range(labels_np.shape[1]):
        column_data = labels_np[:, col]

        # Find where the values change (0->1 or 1->0)
        diffs = np.diff(column_data, prepend=0, append=0)
        run_starts = np.where(diffs == 1)[0]
        run_ends = np.where(diffs == -1)[0]

        # Calculate lengths of each anomaly range
        lengths = run_ends - run_starts
        total_ranges = len(lengths)
        total_anomalies = np.sum(lengths)

        results.append(
            {
                "column": col,
                "total_ranges": total_ranges,
                "total_anomalies": total_anomalies,
                "range_lengths": lengths.tolist(),
                "start_times": run_starts.tolist(),
                "end_times": run_ends.tolist(),
            }
        )

    return results
