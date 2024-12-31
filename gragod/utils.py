import logging
import os
import sys
from pathlib import Path

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


def load_checkpoint_path(checkpoint_folder: str, experiment_name: str) -> Path:
    """
    Load the checkpoint path from the checkpoint folder.
    If the checkpoint folder ends with ".ckpt", it is used as the checkpoint path.
    Otherwise, the checkpoint path is the best.ckpt file in the checkpoint folder.

    If the checkpoint path does not exist, it tries to load the checkpoint from the
    checkpoint folder with the experiment name.

    Args:
        checkpoint_folder: The folder containing the checkpoint.
        experiment_name: The name of the experiment.

    Returns:
        The checkpoint path.
    """
    checkpoint_path = os.path.join(checkpoint_folder, "best.ckpt")

    if not os.path.exists(checkpoint_path):
        checkpoint_path_candidate = os.path.join(
            checkpoint_folder,
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
