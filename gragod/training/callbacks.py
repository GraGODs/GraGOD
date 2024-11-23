from typing import Literal

from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)


def get_training_callbacks(
    log_dir: str,
    model_name: str,
    monitor: str = "Loss/val",
    monitor_mode: Literal["min", "max"] = "min",
    early_stop_patience: int = 10,
    early_stop_delta: float = 0.0001,
    save_top_k: int = 1,
) -> dict[str, Callback]:
    """Creates common callbacks used for training models.

    Args:
        log_dir: Directory to save the checkpoint
        model_name: Name of the model for checkpoint filename
        monitor: Metric to monitor for early stopping and checkpointing
        monitor_mode: Whether to minimize or maximize the monitored metric
        early_stop_patience: Number of epochs to wait before early stopping
        early_stop_delta: Minimum change in monitored value to qualify as an improvement
        save_top_k: Number of best models to save

    Returns:
        Dictionary of callbacks for training
    """
    # Early stopping callback
    early_stop = EarlyStopping(
        monitor=monitor,
        min_delta=early_stop_delta,
        patience=early_stop_patience,
        verbose=True,
        mode=monitor_mode,
    )

    # Model checkpoint callback
    checkpoint = ModelCheckpoint(
        monitor=monitor,
        dirpath=log_dir,
        filename=model_name,
        save_top_k=save_top_k,
        mode=monitor_mode,
    )

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

    return {
        "early_stop": early_stop,
        "checkpoint": checkpoint,
        "lr_monitor": lr_monitor,
    }
