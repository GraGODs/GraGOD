import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.dataloader
from pytorch_lightning.callbacks import ModelCheckpoint

from models.mtad_gat.modules import (
    ConvLayer,
    FeatureAttentionLayer,
    Forecasting_Model,
    GRULayer,
    ReconstructionModel,
    TemporalAttentionLayer,
)


class MTAD_GAT(nn.Module):
    """
    MTAD-GAT model class.
        Args:
            n_features: Number of input features
            window_size: Length of the input sequence
            out_dim: Number of features to output
            kernel_size: size of kernel to use in the 1-D convolution
            feat_gat_embed_dim: embedding dimension (output dimension of linear
                transformation)
                in feat-oriented GAT layer
            time_gat_embed_dim: embedding dimension (output dimension of linear
                transformation)
                in time-oriented GAT layer
            use_gatv2: whether to use the modified attention mechanism of GATv2 instead
                of standard GAT
            gru_n_layers: number of layers in the GRU layer
            gru_hid_dim: hidden dimension in the GRU layer
            forecast_n_layers: number of layers in the FC-based Forecasting Model
            forecast_hid_dim: hidden dimension in the FC-based Forecasting Model
            recon_n_layers: number of layers in the GRU-based Reconstruction Model
            recon_hid_dim: hidden dimension in the GRU-based Reconstruction Model
            dropout: dropout rate
            alpha: negative slope used in the leaky rely activation function
    """

    def __init__(
        self,
        n_features,
        out_dim,
        window_size,
        kernel_size=7,
        use_gatv2=True,
        feat_gat_embed_dim=None,
        time_gat_embed_dim=None,
        gru_n_layers=1,
        gru_hid_dim=300,
        forecast_n_layers=3,
        forecast_hid_dim=300,
        recon_n_layers=1,
        recon_hid_dim=300,
        dropout=0.3,
        alpha=0.2,
    ):
        super(MTAD_GAT, self).__init__()

        self.conv = ConvLayer(n_features, kernel_size)
        self.feature_gat = FeatureAttentionLayer(
            n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2
        )
        self.temporal_gat = TemporalAttentionLayer(
            n_features, window_size, dropout, alpha, time_gat_embed_dim, use_gatv2
        )
        self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(
            gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout
        )
        self.recon_model = ReconstructionModel(
            window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers, dropout
        )

    def forward(self, x):
        """
        Model forward pass.

        Args:
            x: input tensor of shape (b, n, k):
                b - batch size, n - window size, k - number of features
        returns:
            - Predictions tensor of shape (b, out_dim)
            - Reconstruction tensor of shape (b, n, out_dim)
        """
        x = self.conv(x)
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)

        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)

        _, h_end = self.gru(h_cat)
        h_end = h_end.view(x.shape[0], -1)  # Hidden state for last timestamp

        predictions = self.forecasting_model(h_end)
        recons = self.recon_model(h_end)

        return predictions, recons


class MTAD_GAT_PLModule(pl.LightningModule):
    """
    PyTorch Lightning module for the MTAD-GAT model.

    This module encapsulates the MTAD-GAT model and defines the training, validation,
    and optimization procedures using PyTorch Lightning.

    Args:
        model: The MTAD-GAT model instance.
        model_params: Dictionary containing model parameters.
        target_dims: The target dimensions to predict. If None, predict all dimensions.
        init_lr: Initial learning rate for the optimizer.
        forecast_criterion: Loss function for forecasting.
        recon_criterion: Loss function for reconstruction.
        checkpoint_cb: ModelCheckpoint callback for saving best models.
    """

    def __init__(
        self,
        model: nn.Module,
        model_params: dict,
        criterion: dict[str, torch.nn.Module],
        init_lr: float,
        checkpoint_cb: ModelCheckpoint | None = None,
        target_dims: int | None = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.model = model

        self.model_params = model_params
        self.init_lr = init_lr
        self.forecast_criterion = criterion["forecast"]
        self.recon_criterion = criterion["recon"]
        self.target_dims = target_dims
        self.best_model_score = None
        self.checkpoint_cb = checkpoint_cb
        self.best_metrics = None

        self.save_hyperparameters(ignore=["model"])

    def _register_best_metrics(self):
        if self.global_step != 0:
            self.best_metrics = {
                "epoch": self.trainer.current_epoch,
                "train_loss": self.trainer.callback_metrics["Total_loss/train"],
                "train_recon_loss": self.trainer.callback_metrics["Recon_loss/train"],
                "train_forecast_loss": self.trainer.callback_metrics[
                    "Forecast_loss/train"
                ],
                "val_loss": self.trainer.callback_metrics["Total_loss/val"],
                "val_recon_loss": self.trainer.callback_metrics["Recon_loss/val"],
                "val_forecast_loss": self.trainer.callback_metrics["Forecast_loss/val"],
            }

    def forward(self, x):
        return self.model(x)

    def call_logger(
        self,
        loss: torch.Tensor,
        recon_loss: torch.Tensor,
        forecast_loss: torch.Tensor,
        step_type: str,
    ):
        self.log(
            f"Recon_loss/{step_type}",
            recon_loss,
            prog_bar=False,
            on_epoch=True,
            on_step=True,
            logger=True,
        )
        self.log(
            f"Forecast_loss/{step_type}",
            forecast_loss,
            prog_bar=False,
            on_epoch=True,
            on_step=True,
            logger=True,
        )
        self.log(
            f"Total_loss/{step_type}",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            logger=True,
        )

    def shared_step(self, batch, batch_idx):
        x, y, *_ = batch
        preds, recons = self(x)

        if self.target_dims is not None:
            x = x[:, :, self.target_dims]
            y = y[:, :, self.target_dims].squeeze(-1)
            preds = preds[..., self.target_dims].squeeze(-1)
            recons = recons[..., self.target_dims].squeeze(-1)

        if preds.ndim == 3:
            preds = preds.squeeze(1)
        if y.ndim == 3:
            y = y.squeeze(1)

        forecast_loss = torch.sqrt(self.forecast_criterion(y, preds))
        recon_loss = torch.sqrt(self.recon_criterion(x, recons))
        loss = forecast_loss + recon_loss

        return loss, recon_loss, forecast_loss

    def training_step(self, batch, batch_idx):
        loss, recon_loss, forecast_loss = self.shared_step(batch, batch_idx)
        self.call_logger(loss, recon_loss, forecast_loss, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon_loss, forecast_loss = self.shared_step(batch, batch_idx)
        self.call_logger(loss, recon_loss, forecast_loss, "val")
        return loss

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)  # type: ignore
        return optimizer
