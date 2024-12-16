from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.dataloader
from lightning.pytorch.utilities import grad_norm
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from gragod import PathType

# TODO:
# - Add SWA


class PLBaseModule(pl.LightningModule):
    """
    Base class for PyTorch Lightning modules.

    Every model that inherits from this class will have the same configuration
    for the optimizer.
    """

    def __init__(
        self,
        model: nn.Module,
        model_params: dict,
        checkpoint_cb: ModelCheckpoint,
        init_lr: float = 0.001,
        criterion: torch.nn.Module | dict[str, torch.nn.Module] = nn.MSELoss(),
        target_dims: int | None = None,
        weight_decay: float = 1e-5,
        eps: float = 1e-8,
        betas: tuple[float, float] = (0.9, 0.999),
        gradient_clip_val: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.model_params = model_params
        self.init_lr = init_lr
        self.checkpoint_cb = checkpoint_cb
        self.target_dims = target_dims
        self.best_model_score = None
        self.best_metrics = None
        self.weight_decay = weight_decay
        self.eps = eps
        self.betas = betas
        self.gradient_clip_val = gradient_clip_val

        if isinstance(criterion, nn.Module):
            self.criterion = criterion
        elif isinstance(criterion, dict):
            for key, value in criterion.items():
                setattr(self, key, value)
        else:
            raise ValueError(
                "Criterion must be a nn.Module or a dictionary of nn.Module"
                f"Criterion: {criterion}"
            )

        self.save_hyperparameters(ignore=["model", "criterion"])

    def _register_best_metrics(self):
        """
        Register the best metrics. Defined in the child class.
        """
        pass

    def on_train_epoch_start(self):
        if (
            self.checkpoint_cb is not None
            and self.checkpoint_cb.best_model_score is not None
        ):
            if self.best_model_score is None:
                self.best_model_score = float(self.checkpoint_cb.best_model_score)
                self._register_best_metrics()
            elif (
                self.checkpoint_cb.mode == "min"
                and float(self.checkpoint_cb.best_model_score) < self.best_model_score
            ) or (
                self.checkpoint_cb.mode == "max"
                and float(self.checkpoint_cb.best_model_score) > self.best_model_score
            ):
                self.best_model_score = float(self.checkpoint_cb.best_model_score)
                self._register_best_metrics()

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(  # type: ignore
            self.parameters(),
            lr=self.init_lr,
            weight_decay=self.weight_decay,
            eps=self.eps,
            betas=self.betas,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8, verbose=True  # type: ignore
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "Loss/val",
                "interval": "epoch",
            },
        }


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
        model_pl: type[PLBaseModule],
        model_params: dict,
        criterion: torch.nn.Module | dict[str, torch.nn.Module],
        batch_size: int,
        n_epochs: int,
        init_lr: float,
        device: str,
        log_dir: str,
        logger: TensorBoardLogger,
        callbacks: list[pl.Callback],
        checkpoint_cb: ModelCheckpoint,
        target_dims: int | None = None,
        log_every_n_steps: int = 1,
        weight_decay: float = 1e-5,
        eps: float = 1e-8,
        betas: tuple[float, float] = (0.9, 0.999),
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
            weight_decay=weight_decay,
            eps=eps,
            betas=betas,
        )

        self.logger = logger

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader | None = None,
        args_summary: dict = {},
    ):
        self.trainer = pl.Trainer(
            max_epochs=self.n_epochs,
            accelerator=self.device,
            logger=self.logger,
            log_every_n_steps=self.log_every_n_steps,
            callbacks=self.callbacks,
            gradient_clip_val=1.0,
        )

        self.trainer.fit(self.lightning_module, train_loader, val_loader)
        best_metrics = {
            k: v for k, v in self.lightning_module.best_metrics.items()  # type: ignore
        }
        print(f"BEST METRICS: {best_metrics}")
        self.logger.log_hyperparams(params=args_summary, metrics=best_metrics)

    def predict(self, loader: torch.utils.data.DataLoader):
        return self.trainer.predict(self.lightning_module, loader)
