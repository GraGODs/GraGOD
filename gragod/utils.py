import logging
import os
import sys
from pathlib import Path

import torch

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


def load_checkpoint_path(checkpoint_folder: PathType, experiment_name: str) -> Path:
    checkpoint_path = os.path.join(checkpoint_folder, "best.ckpt")

    if not os.path.exists(checkpoint_path):
        print(
            f"Checkpoint not found at {checkpoint_path}. Trying with {experiment_name}"
        )
        checkpoint_path = os.path.join(
            checkpoint_folder,
            f"{experiment_name}.ckpt",
        )
        if not os.path.exists(checkpoint_path):
            raise ValueError(
                f"Checkpoint not found at {checkpoint_path}. "
                f"Tried with {experiment_name}"
            )

    if not checkpoint_path.endswith(".ckpt"):
        raise ValueError(f"Checkpoint must end with .ckpt, got {checkpoint_path}")

    return Path(checkpoint_path)


def jit_compile_model(input_example: torch.Tensor, model, save_dir: PathType):
    with torch.jit.optimized_execution(True):
        traced_model = torch.jit.trace(model, input_example)

    print(f"Saving model in {save_dir}")
    torch.jit.save(traced_model, save_dir)


def pytest_is_running():
    return any(arg.startswith("pytest") for arg in sys.argv)
