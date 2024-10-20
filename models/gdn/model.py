import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gdn.modules import GNNLayer, OutLayer


class GDN(nn.Module):
    """
    Graph Deviation Network (GDN) model.

    This model uses graph neural networks to detect deviations in graph-structured data.

    Attributes:
        edge_index_sets (list): List of edge indices for different graph structures.
        embedding (nn.Embedding): Node embedding layer.
        bn_outlayer_in (nn.BatchNorm1d): Batch normalization layer for output.
        gnn_layers (nn.ModuleList): List of GNN layers.
        topk (int): Number of top similarities to consider for each node.
        learned_graph (torch.Tensor): Learned graph structure.
        out_layer (OutLayer): Output layer.
        cache_edge_index_sets (list): Cached edge indices for batch processing.
        dp (nn.Dropout): Dropout layer.

    Args:
        edge_index_sets (list): List of edge indices for different graph structures.
        node_num (int): Number of nodes in the graph.
        dim (int, optional): Dimension of node embeddings. Defaults to 64.
        out_layer_inter_dim (int, optional): Intermediate dimension in output layer. Defaults to 256.
        input_dim (int, optional): Input feature dimension. Defaults to 10.
        out_layer_num (int, optional): Number of layers in output MLP. Defaults to 1.
        topk (int, optional): Number of top similarities to consider for each node. Defaults to 20.
    """

    def __init__(
        self,
        edge_index_sets: list[torch.Tensor],
        node_num: int,
        dim: int = 64,
        out_layer_inter_dim: int = 256,
        input_dim: int = 10,
        out_layer_num: int = 1,
        topk: int = 20,
    ):

        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets

        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList(
            [GNNLayer(input_dim, dim, heads=1) for i in range(edge_set_num)]
        )

        self.topk = topk
        self.learned_graph = None

        self.out_layer = OutLayer(
            dim * edge_set_num, out_layer_num, inter_num=out_layer_inter_dim
        )

        self.cache_edge_index_sets = [None] * edge_set_num

        self.dp = nn.Dropout(0.2)

        self.init_params()

    def init_params(self):
        """Initialize model parameters."""
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GDN model.

        Args:
            data (torch.Tensor): Input data tensor of shape [batch_size, node_num, feature_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size * node_num].
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
                cache_edge_index is None
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

            topk_num = self.topk

            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

            self.learned_graph = topk_indices_ji

            gated_i = (
                torch.arange(0, node_num)
                .unsqueeze(1)
                .repeat(1, topk_num)
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
        out = torch.mul(x, self.embedding(indexes))

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
            org_edge_index (torch.Tensor): Original edge index.
            batch_num (int): Number of graphs in the batch.
            node_num (int): Number of nodes in each graph.

        Returns:
            torch.Tensor: Batched edge index.
        """
        edge_index = org_edge_index.clone().detach()
        edge_num = org_edge_index.shape[1]
        batch_edge_index = edge_index.repeat(1, batch_num).contiguous()

        for i in range(batch_num):
            batch_edge_index[:, i * edge_num : (i + 1) * edge_num] += i * node_num

        return batch_edge_index.long()
