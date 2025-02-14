from typing import Literal

import torch
import torch.utils.data
import torch.utils.data.dataloader
from torch.nn import Linear
from torch_geometric.nn import TAGConv

from gragod.training.trainer import PLBaseModule


class GCN(torch.nn.Module):
    """Graph Convolutional Network for time series forecasting.

    Args:
        window_size: Size of the sliding window
        n_layers: Number of graph convolutional layers
        hidden_dim: Dimension of hidden layers
        k: Number of hops to consider in TAGConv
    """

    def __init__(
        self,
        window_size: int = 5,
        n_layers: int = 3,
        hidden_dim: int = 32,
        K: int = 1,
        **kwargs,
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


class GCN_PLModule(PLBaseModule):
    """
    PyTorch Lightning module for the GCN model.

    Args:
        model: The GCN model instance
        model_params: Dictionary containing model parameters
        init_lr: Initial learning rate for the optimizer
        criterion: Loss function for training
        checkpoint_cb: ModelCheckpoint callback for saving best models
    """

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

    def predict_step(self, batch, batch_idx):
        """
        Prediction step for the model.

        Args:
            batch: The input batch from the dataloader
            batch_idx: The index of the current batch

        Returns:
            tuple: predictions
        """
        x, _, _, edge_index = batch
        predictions, _ = self(x, edge_index)
        return predictions

    def post_process_predictions(self, predictions):
        """Post-process the predictions."""
        predictions = torch.cat(predictions)
        predictions = predictions[:-1, :]
        return predictions

    def calculate_anomaly_score(
        self,
        predict_output: torch.Tensor,
        X_true: torch.Tensor,
        score_type: Literal["abs", "sqr"],
        **kwargs,
    ):
        """Calculate the anomaly score."""
        predictions = self.post_process_predictions(predict_output)
        X_true = X_true[:-1, :]
        if score_type == "abs":
            return torch.abs(predictions - X_true)
        elif score_type == "sqr":
            return torch.sqrt((predictions - X_true) ** 2)
        else:
            raise ValueError(f"Invalid score type: {score_type}")
