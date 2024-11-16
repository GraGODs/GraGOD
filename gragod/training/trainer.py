import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.dataloader
from pytorch_lightning.loggers import TensorBoardLogger

from gragod import PathType


class TrainerPL:
    """
    Trainer class for the MTAD-GAT model using PyTorch Lightning.

    This class sets up the training environment, including callbacks, loggers,
    and the PyTorch Lightning Trainer.

    Args:
        model: The model instance.
        model_params: Dictionary containing model parameters.
        model_pl: PyTorch Lightning module class to use for training.
        criterion: Dictionary containing loss functions.
        n_epochs: Number of training epochs.
        batch_size: Batch size for training and validation.
        init_lr: Initial learning rate for the optimizer.
        device: Device to use for training ('cpu' or 'cuda').
        log_dir: Directory for saving logs and checkpoints.
        callbacks: Additional callbacks for the Trainer.
        log_every_n_steps: Frequency of logging steps.
        target_dims: The target dimensions to focus on. If None, use all dimensions.
    """

    def __init__(
        self,
        # Model related
        model: nn.Module,
        model_pl: type[pl.LightningModule],
        model_params: dict,
        criterion: torch.nn.Module | dict[str, torch.nn.Module],
        batch_size: int,
        n_epochs: int,
        init_lr: float,
        device: str,
        log_dir: str,
        logger: TensorBoardLogger,
        callbacks: list[pl.Callback],
        checkpoint_cb: pl.Callback | None,
        target_dims: int | None = None,
        log_every_n_steps: int = 1,
    ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.log_dir = log_dir
        self.log_every_n_steps = log_every_n_steps

        self.callbacks = callbacks

        self.lightning_module = model_pl(
            model=model,
            model_params=model_params,
            init_lr=init_lr,
            checkpoint_cb=checkpoint_cb,
            criterion=criterion,
            target_dims=target_dims,
        )

        self.logger = logger

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader | None = None,
        args_summary: dict = {},
    ):
        trainer = pl.Trainer(
            max_epochs=self.n_epochs,
            accelerator=self.device,
            logger=self.logger,
            log_every_n_steps=self.log_every_n_steps,
            callbacks=self.callbacks,
        )

        trainer.fit(self.lightning_module, train_loader, val_loader)

        best_metrics = {
            k: v
            for k, v in self.lightning_module.best_metrics.items()  # type: ignore
            if "epoch" in k
        }
        self.logger.log_hyperparams(params=args_summary, metrics=best_metrics)

    def load(self, path: PathType):
        self.lightning_module.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )
