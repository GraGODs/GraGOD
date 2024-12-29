import torch
import torch.nn as nn
from tsai.models.RNN import GRU

from gragod.training.trainer import PLBaseModule


class GRUModel(nn.Module):
    """
    GRU model class. Implements a GRU with a forecasting head.
    Args:
        n_features: Number of input features
        window_size: Length of the input sequence
        out_dim: Number of features to output
        hidden_size: Hidden dimension in the GRU layer
        n_layers: Number of layers in the GRU
        bidirectional: Whether to use bidirectional GRU
        dropout: Dropout rate
        fc_dropout: Dropout rate for fully connected layers
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 300,
        n_layers: int = 3,
        bidirectional: bool = True,
        rnn_dropout: float = 0.0,
        fc_dropout: float = 0.3,
        **kwargs,
    ):
        super(GRUModel, self).__init__()

        # GRU from tsai
        self.gru = GRU(
            c_in=n_features,
            c_out=n_features,
            hidden_size=hidden_size,
            n_layers=n_layers,
            bidirectional=bidirectional,
            rnn_dropout=rnn_dropout,  # type: ignore
            fc_dropout=fc_dropout,
        )

        # # Forecasting head
        # self.forecasting_model = nn.Sequential(
        #     nn.Linear(gru_out_dim, hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_size, out_dim)
        # )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)
        Returns:
            predictions: Tensor of shape (batch_size, out_dim)
        """
        # GRU expects input shape: (batch_size, n_features, seq_len)
        x = x.transpose(1, 2)

        # Get GRU output
        gru_out = self.gru(x)

        # Generate predictions
        # predictions = self.forecasting_model(gru_out)

        return gru_out


class GRU_PLModule(PLBaseModule):
    """
    PyTorch Lightning module for the GRU model.

    This module encapsulates the GRU model and defines the training, validation,
    and optimization procedures using PyTorch Lightning.
    """

    def _register_best_metrics(self):
        if self.global_step != 0:
            self.best_metrics = {
                "epoch": self.trainer.current_epoch,
                "train_loss": float(self.trainer.callback_metrics["Loss/train"]),
                "val_loss": float(self.trainer.callback_metrics["Loss/val"]),
            }

    def call_logger(
        self,
        loss: torch.Tensor,
        step_type: str,
    ):
        self.log(
            f"Loss/{step_type}",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            logger=True,
        )

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, batch_idx):
        x, y, *_ = batch
        gru_out = self(x)

        if self.target_dims is not None:
            x = x[:, :, self.target_dims]
            y = y[:, :, self.target_dims].squeeze(-1)

        if gru_out.ndim == 3:
            gru_out = gru_out.squeeze(1)
        if y.ndim == 3:
            y = y.squeeze(1)

        loss = torch.sqrt(self.forecast_criterion(y, gru_out))
        return loss

    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        loss = self.shared_step(batch, batch_idx)
        self.call_logger(loss, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        loss = self.shared_step(batch, batch_idx)
        self.call_logger(loss, "val")
        return loss

    def predict_step(self, batch, batch_idx):
        """Prediction step for the model."""
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        predictions = self(x)
        return predictions
