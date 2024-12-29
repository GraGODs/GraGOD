import torch
import torch.nn as nn

from gragod.training.trainer import PLBaseModule


class Sampling(nn.Module):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding."""

    def __init__(self):
        super().__init__()

    def forward(self, z_mean, z_log_var):
        batch = z_mean.size(0)
        seq = z_mean.size(1)
        dim = z_mean.size(2)
        epsilon = torch.randn(batch, seq, dim, device=z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class Conv1DBlock(nn.Module):
    """1D Convolutional block with batch normalization"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        stride=1,
        activation=None,
        padding="causal",
    ):
        super().__init__()
        # For causal padding, we only pad on the left side
        self.padding = (kernel_size - 1) * dilation if padding == "causal" else padding
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # We'll handle padding manually
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x, target_size=None):
        # Manual causal padding
        if self.padding > 0:
            padding = torch.zeros(x.shape[0], x.shape[1], self.padding, device=x.device)
            x = torch.cat([padding, x], dim=2)

        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.activation(x)

        # Trim to target size if specified
        if target_size is not None and x.shape[2] > target_size:
            x = x[:, :, -target_size:]

        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size, dilations, latent_dim):
        super().__init__()
        self.sampling = Sampling()

        # Build encoder layers
        self.encoder_layers = nn.ModuleList()
        in_channels = input_dim

        # First layer
        self.encoder_layers.append(
            Conv1DBlock(
                in_channels,
                hidden_dims[0],
                kernel_size,
                dilations[0],
                activation=nn.Tanh(),
            )
        )

        # Middle layers
        for i in range(len(hidden_dims) - 1):
            self.encoder_layers.append(
                Conv1DBlock(
                    hidden_dims[i],
                    hidden_dims[i + 1],
                    kernel_size,
                    dilations[i + 1],
                    activation=nn.Tanh(),
                )
            )

        # Output layers for z_mean and z_log_var
        self.z_mean = Conv1DBlock(
            hidden_dims[-1], latent_dim, kernel_size, dilations[-1]
        )

        self.z_log_var = Conv1DBlock(
            hidden_dims[-1], latent_dim, kernel_size, dilations[-1]
        )

    def forward(self, x):
        # Store input sequence length
        seq_len = x.shape[1]

        # Transpose for 1D convolution (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x, seq_len)

        # Get latent parameters and ensure they match input sequence length
        z_mean = self.z_mean(x, seq_len)
        z_log_var = self.z_log_var(x, seq_len)

        # Sample latent vector
        z = self.sampling(z_mean.transpose(1, 2), z_log_var.transpose(1, 2))

        return z_mean.transpose(1, 2), z_log_var.transpose(1, 2), z


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, kernel_size, dilations, output_dim):
        super().__init__()

        # Build decoder layers
        self.decoder_layers = nn.ModuleList()

        # First layer
        self.decoder_layers.append(
            Conv1DBlock(
                latent_dim,
                hidden_dims[-1],
                kernel_size,
                dilations[-1],
                activation=nn.ELU(),
            )
        )

        # Middle layers
        for i in range(len(hidden_dims) - 1, 0, -1):
            self.decoder_layers.append(
                Conv1DBlock(
                    hidden_dims[i],
                    hidden_dims[i - 1],
                    kernel_size,
                    dilations[i - 1],
                    activation=nn.ELU(),
                )
            )

        # Output layers
        self.x_mean = Conv1DBlock(hidden_dims[0], output_dim, kernel_size, dilations[0])
        self.x_log_var = Conv1DBlock(
            hidden_dims[0], output_dim, kernel_size, dilations[0]
        )

    def forward(self, z):
        # Store input sequence length
        seq_len = z.shape[1]

        # Transpose for 1D convolution
        x = z.transpose(1, 2)

        # Decoder layers
        for layer in self.decoder_layers:
            x = layer(x, seq_len)

        # Get output parameters and ensure they match input sequence length
        x_mean = self.x_mean(x, seq_len)
        x_log_var = self.x_log_var(x, seq_len)

        return x_mean.transpose(1, 2), x_log_var.transpose(1, 2)


class DCVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims=[32, 16, 1],
        kernel_size=2,
        dilations=[1, 8, 16],
        latent_dim=1,
    ):
        super().__init__()

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            dilations=dilations,
            latent_dim=latent_dim,
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            dilations=dilations,
            output_dim=input_dim,
        )

    def forward(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        x_mean, x_log_var = self.decoder(z)
        return x_mean, x_log_var, z_mean, z_log_var


class DCVAE_PLModule(PLBaseModule):
    def _register_best_metrics(self):
        if self.global_step != 0:
            self.best_metrics = {
                "epoch": self.trainer.current_epoch,
                "train_loss": float(self.trainer.callback_metrics["Loss/train"]),
                "train_recon_loss": float(
                    self.trainer.callback_metrics["Recon_loss/train"]
                ),
                "train_kl_loss": float(self.trainer.callback_metrics["KL_loss/train"]),
                "val_loss": float(self.trainer.callback_metrics["Loss/val"]),
                "val_recon_loss": float(
                    self.trainer.callback_metrics["Recon_loss/val"]
                ),
                "val_kl_loss": float(self.trainer.callback_metrics["KL_loss/val"]),
            }

    def call_logger(
        self,
        loss: torch.Tensor,
        kl_loss: torch.Tensor,
        recon_loss: torch.Tensor,
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
        self.log(
            f"KL_loss/{step_type}",
            kl_loss,
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            logger=True,
        )
        self.log(
            f"Recon_loss/{step_type}",
            recon_loss,
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            logger=True,
        )

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, batch_idx):
        x, *_ = batch
        x_mean, x_log_var, z_mean, z_log_var = self(x)

        # Reconstruction loss (first term in equation)
        # (x - μ)²/σ² + log(σ²)
        recon_term1 = torch.square(
            (x - x_mean) / torch.exp(0.5 * x_log_var)
        )  # (x - μ)²/σ²
        recon_term2 = x_log_var  # log(σ²)
        reconstruction_loss = 0.5 * torch.mean(recon_term1 + recon_term2)

        # KL loss (second term in equation)
        # -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.mean(
            torch.sum(
                1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var), dim=-1
            )
        )

        # Total loss with proper scaling
        loss = reconstruction_loss + kl_loss
        return loss, reconstruction_loss, kl_loss

    def training_step(self, batch, batch_idx):
        loss, recon_loss, kl_loss = self.shared_step(batch, batch_idx)
        self.call_logger(loss, kl_loss, recon_loss, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon_loss, kl_loss = self.shared_step(batch, batch_idx)
        self.call_logger(loss, kl_loss, recon_loss, "val")
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x_mean, x_log_var, z_mean, z_log_var = self(x)
        return x_mean, x_log_var
