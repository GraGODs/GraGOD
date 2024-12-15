import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.dataloader

from gragod.training.trainer import PLBaseModule
from models.gdn.modules import GNNLayer, OutLayer


class GDN(nn.Module):
    """
    Graph Deviation Network (GDN) model.

    This model uses graph neural networks to detect deviations in graph-structured data

    Attributes:
        edge_index_sets: List of edge indices for different graph structures.
        embedding: Node embedding layer.
        bn_outlayer_in: Batch normalization layer for output.
        gnn_layers: List of GNN layers.
        topk: Number of top similarities to consider for each node.
        learned_graph: Learned graph structure.
        out_layer: Output layer.
        cache_edge_index_sets: Cached edge indices for batch processing.
        dp: Dropout layer.

    Args:
        edge_index_sets: List of edge indices for different graph structures.
        node_num: Number of nodes in the graph.
        embed_dim: Dimension of node embeddings.
        out_layer_inter_dim: Intermediate dimension in output layer.
        window_size: Input feature dimension.
        out_layer_num: Number of layers in output MLP.
        topk: Number of top similarities to consider for each node.
        heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        edge_index_sets: list[torch.Tensor],
        node_num: int,
        embed_dim: int = 64,
        out_layer_inter_dim: int = 256,
        window_size: int = 10,
        out_layer_num: int = 1,
        topk: int = 20,
        heads: int = 1,
        dropout: float = 0,
        negative_slope: float = 0.2,
    ):
        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets

        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(heads * embed_dim)

        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList(
            [
                GNNLayer(
                    window_size,
                    embed_dim,
                    heads=heads,
                    dropout=dropout,
                    negative_slope=negative_slope,
                )
                for _ in range(edge_set_num)
            ]
        )

        self.topk = topk
        self.learned_graph = None
        self.heads = heads

        self.out_layer = OutLayer(
            embed_dim * heads * edge_set_num,
            out_layer_num,
            inter_num=out_layer_inter_dim,
        )

        self.cache_edge_index_sets = [torch.tensor([]) for _ in range(edge_set_num)]

        self.dp = nn.Dropout(dropout)
        self.init_params()

    def init_params(self):
        """Initialize model parameters."""
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GDN model.

        Args:
            data: Input data tensor of shape [batch_size, node_num, feature_dim].

        Returns:
            Output tensor of shape [batch_size * node_num].
        """

        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets

        device = data.device

        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()

        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]
            if (
                cache_edge_index.nelement() == 0
                or cache_edge_index.shape[1] != edge_num * batch_num
            ):
                self.cache_edge_index_sets[i] = self._get_batch_edge_index(
                    edge_index, batch_num, node_num
                ).to(device)

            all_embeddings = self.embedding(torch.arange(node_num).to(device))

            weights_arr = all_embeddings.detach().clone()
            all_embeddings = all_embeddings.repeat(batch_num, 1)

            weights = weights_arr.view(node_num, -1)

            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(
                weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1)
            )
            cos_ji_mat = cos_ji_mat / normed_mat

            topk_indices_ji = torch.topk(cos_ji_mat, self.topk, dim=-1)[1]

            self.learned_graph = topk_indices_ji

            gated_i = (
                torch.arange(0, node_num)
                .unsqueeze(1)
                .repeat(1, self.topk)
                .flatten()
                .to(device)
                .unsqueeze(0)
            )
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

            batch_gated_edge_index = self._get_batch_edge_index(
                gated_edge_index, batch_num, node_num
            ).to(device)
            gcn_out = self.gnn_layers[i](
                x,
                batch_gated_edge_index,
                node_num=node_num * batch_num,
                embedding=all_embeddings,
            )

            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)

        indexes = torch.arange(0, node_num).to(device)
        node_embeddings = self.embedding(indexes)

        batch_size, node_num, hidden_dim = x.shape
        heads = self.heads
        embed_dim = hidden_dim // heads
        x_reshaped = x.view(batch_size, node_num, heads, embed_dim)

        embeddings_expanded = node_embeddings.view(node_num, 1, embed_dim).expand(
            -1, heads, -1
        )

        out = torch.mul(x_reshaped, embeddings_expanded.unsqueeze(0))
        out = out.view(batch_size, node_num, heads * embed_dim)

        out = out.permute(0, 2, 1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)

        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, node_num)

        return out

    def _get_batch_edge_index(
        self, org_edge_index: torch.Tensor, batch_num: int, node_num: int
    ) -> torch.Tensor:
        """
        Get batched edge index for multiple graphs.

        Args:
            org_edge_index: Original edge index.
            batch_num: Number of graphs in the batch.
            node_num: Number of nodes in each graph.

        Returns:
            Batched edge index.
        """
        edge_index = org_edge_index.clone().detach()
        edge_num = org_edge_index.shape[1]
        batch_edge_index = edge_index.repeat(1, batch_num).contiguous()

        for i in range(batch_num):
            batch_edge_index[:, i * edge_num : (i + 1) * edge_num] += i * node_num

        return batch_edge_index.long()


class GDN_PLModule(PLBaseModule):
    """
    PyTorch Lightning module for the GDN model.

    This module encapsulates the GDN model and defines the training, validation,
    and optimization procedures using PyTorch Lightning.

    Args:
        model: The GDN model instance
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

    def forward(self, x):
        """Forward pass of the model."""
        return self.model(x)

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
                x.reshape(-1, x.size(2), x.size(1)),
                y.squeeze(1),
                edge_index,
            ]
        ]
        out = self(x)
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
            predictions: Predictions for the input batch
        """
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        predictions = self(x.reshape(-1, x.size(2), x.size(1)))
        return predictions
