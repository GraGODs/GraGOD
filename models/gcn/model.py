from typing import Any

import pytorch_lightning as pl
import torch
import torch.utils.data
import torch.utils.data.dataloader
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import TAGConv


class GCN(torch.nn.Module):
    """Graph Convolutional Network for time series forecasting.

    Args:
        window_size: Size of the sliding window
        n_layers: Number of graph convolutional layers
        hidden_dim: Dimension of hidden layers
        k: Number of hops to consider in TAGConv
    """

    def __init__(
        self, window_size: int = 5, n_layers: int = 3, hidden_dim: int = 32, K: int = 1
    ):
        super(GCN, self).__init__()
        self.window_size = window_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.K = K
        self.conv_layers = torch.nn.ModuleList(
            [TAGConv(window_size, hidden_dim, K=K)]
            + [TAGConv(hidden_dim, hidden_dim, K=K) for _ in range(n_layers - 1)]
        )
        self.tanh = torch.nn.Tanh()
        self.regressor = Linear(hidden_dim, 1)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor):
        """Forward pass of the model.

        Args:
            X: Input tensor of shape (batch_size, window_size, num_nodes)
            edge_index: Graph connectivity in COO format of shape (2, num_edges)

        Returns:
            tuple: (predictions, hidden_states)
                - predictions: Tensor of shape (batch_size, 1, num_nodes)
                - hidden_states: Tensor of shape (batch_size * num_nodes, hidden_dim)
        """
        batch_size = X.size(0)
        num_nodes = X.size(2)
        # Reshape to [batch_size * num_nodes, window_size]
        h = X.reshape(-1, self.window_size)

        for conv in self.conv_layers:
            # Create batch-wise edge indices by adding appropriate offsets
            batch_size, _, num_edges = edge_index.shape

            offset = (
                torch.arange(batch_size, device=edge_index.device).view(-1, 1, 1)
                * num_nodes
            ).repeat(1, 2, num_edges)

            batch_edge_index = (
                (edge_index.long() + offset.long()).permute(1, 0, 2).reshape(2, -1)
            )

            h = conv(h, batch_edge_index)
            h = self.tanh(h)

        out = self.regressor(h)
        out = out.reshape(batch_size, num_nodes)

        return out, h

    def detect_anomalies(
        self, X: torch.Tensor, predictions: torch.Tensor, threshold: float = 0.01
    ):
        """Detect anomalies in the data.

        An anomaly is classified as such if the absolute difference between the
        prediction and the actual value is greater than the threshold.

        Args:
            X: Input tensor of shape (data_length, num_nodes)
            predictions: Predictions tensor of shape (data_length, num_nodes)
            threshold: Threshold for the anomaly score

        Returns:
            Anomaly scores tensor of shape (data_length, num_nodes)
        """
        diff = torch.abs(X - predictions)
        anomalies = diff > threshold
        return anomalies


class GCN_PLModule(pl.LightningModule):
    """
    PyTorch Lightning module for the GCN model.

    Args:
        model: The GCN model instance
        model_params: Dictionary containing model parameters
        init_lr: Initial learning rate for the optimizer
        criterion: Loss function for training
        checkpoint_cb: ModelCheckpoint callback for saving best models
    """

    def __init__(
        self,
        model: nn.Module,
        model_params: dict,
        init_lr: float = 0.001,
        criterion: torch.nn.Module = nn.MSELoss(),
        checkpoint_cb: ModelCheckpoint | None = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.model_params = model_params
        self.init_lr = init_lr
        self.criterion = criterion
        self.checkpoint_cb = checkpoint_cb
        self.best_model_score = None
        self.best_metrics = None

        self.save_hyperparameters(ignore=["model"])

    def _register_best_metrics(self):
        """Register the best metrics during training."""
        if self.global_step != 0:
            self.best_metrics = {
                "epoch": self.trainer.current_epoch,
                "train_loss": self.trainer.callback_metrics["Loss/train"],
                "val_loss": self.trainer.callback_metrics["Loss/val"],
            }

    def forward(self, x, edge_index):
        """Forward pass of the model."""
        return self.model(x, edge_index)

    def call_logger(self, loss: torch.Tensor, step_type: str):
        """Log metrics during training/validation."""
        self.log(
            f"Loss/{step_type}",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            logger=True,
        )

    def shared_step(self, batch, batch_idx):
        """Shared step for both training and validation."""
        x, y, _, edge_index = batch
        x, y, edge_index = [
            item.float().to(self.device)
            for item in [
                x,
                y.squeeze(1),
                edge_index,
            ]
        ]
        out, _ = self(x, edge_index)
        loss = self.criterion(out, y)
        return loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss = self.shared_step(batch, batch_idx)
        self.call_logger(loss, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss = self.shared_step(batch, batch_idx)
        self.call_logger(loss, "val")
        return loss

    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
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
        """Configure optimizers for training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)  # type: ignore
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, verbose=True  # type: ignore
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "Loss/val",
                "interval": "epoch",
            },
        }

    def predict_step(self, batch, batch_idx):
        """
        Prediction step for the model.

        Args:
            batch: The input batch from the dataloader
            batch_idx: The index of the current batch

        Returns:
            tuple: (predictions, reconstructions)
        """
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        predictions, _ = self(x)
        return predictions
