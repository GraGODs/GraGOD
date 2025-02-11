import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax


class OutLayer(nn.Module):
    """
    Output layer for the GNN model.

    This layer consists of a series of linear layers, optionally
    followed by batch normalization and ReLU activation.
    The final layer always outputs a single value.

    Attributes:
        mlp: A list of modules representing the layers of the MLP.

    Args:
        in_num: The number of input features.
        layer_num: The number of layers in the MLP.
        inter_num: The number of neurons in intermediate layers.
    """

    def __init__(self, in_num, layer_num, inter_num=512):
        super(OutLayer, self).__init__()

        modules = []
        for i in range(layer_num - 1):
            layer_in_num = in_num if i == 0 else inter_num
            modules.append(nn.Linear(layer_in_num, inter_num))
            modules.append(nn.BatchNorm1d(inter_num))
            modules.append(nn.ReLU())

        # last layer, output shape:1
        modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, 1))

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        """
        Forward pass of the OutLayer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through the MLP.
        """
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)

        return out


class GNNLayer(nn.Module):
    """
    A Graph Neural Network layer.

    This layer applies a graph convolution followed by batch normalization
    and ReLU activation.

    Attributes:
        gnn: The graph convolutional layer.
        bn: Batch normalization layer.
        relu: ReLU activation function.

    Args:
        in_channel: Number of input channels.
        out_channel: Number of output channels.
        heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        heads: int = 1,
        dropout: float = 0,
        negative_slope: float = 0.2,
    ):
        super(GNNLayer, self).__init__()

        self.gnn = GraphLayer(
            in_channel,
            out_channel,
            heads=heads,
            dropout=dropout,
            negative_slope=negative_slope,
        )

        self.bn = nn.BatchNorm1d(heads * out_channel)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, embedding=None):
        """
        Forward pass of the GNNLayer.

        Args:
            x: Input node features.
            edge_index: Graph connectivity in COO format.
            embedding: Node embeddings. Defaults to None.
            node_num: Number of nodes. Defaults to 0.

        Returns:
            Output tensor after applying graph convolution,
            batch normalization, and ReLU.
        """
        out = self.gnn(x, edge_index, embedding)
        out = self.bn(out)
        return self.relu(out)


class GraphLayer(MessagePassing):
    """
    Class for graph convolutional layers using message passing.

    Attributes:
        in_channels: Number of input channels for the layer
        out_channels: Number of output channels for the layer
        heads: Number of heads for multi-head attention
        negative_slope: Slope for LeakyReLU
        dropout: Dropout rate
        lin: Linear layer for transforming input
        att_i: Attention parameter related to x_i
        att_j: Attention parameter related to x_j
        att_em_i: Attention parameter related to embedding of x_i
        att_em_j: Attention parameter related to embedding of x_j
        bias: Bias parameter added after message propagation

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        heads: Number of attention heads.
        negative_slope: Negative slope for LeakyReLU.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        negative_slope: float = 0.2,
        dropout: float = 0,
    ):
        super(GraphLayer, self).__init__(aggr="add", node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        # output channels are heads * out_channels
        self._out_channels = heads * out_channels

        # parameters related to weight matrix W
        self.lin = Linear(in_channels, self._out_channels, bias=False)

        self.bias = Parameter(torch.Tensor(self._out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialise parameters of GraphLayer."""
        glorot(self.lin.weight)
        self.bias.data.zero_()

    def forward(self, x, edge_index, embedding):
        """Forward method for propagating messages of GraphLayer.

        Args:
            x: Node features tensor of shape
                [N x batch_size, in_channels], where N is the number of nodes.

            edge_index: Graph connectivity in COO format,
                shape [2, E x batch_size], where E is the number of edges.

            embedding: Node embeddings tensor of shape
                [N x batch_size, out_channels].

        Returns:
            Output tensor after message passing and attention mechanism.
        """
        # linearly transform node feature matrix
        assert torch.is_tensor(x)
        x = self.lin(x)

        # add self loops, nodes are in dim 0 of x
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(self.node_dim))

        # propagate messages
        out = self.propagate(
            edge_index,
            x=(x, x),
            embedding=embedding,
            edges=edge_index,
        )

        # transform [N x batch_size, 1, _out_channels]
        # to [N x batch_size, _out_channels]
        out = out.view(-1, self._out_channels)

        # apply final bias vector
        out += self.bias

        return out

    def message(self, x_i, x_j, edge_index_i, size_i, embedding, edges):
        """Calculate the attention weights using cosine similarity between embeddings.

        Args:
            x_i: Source node features of shape [(topk x N x batch_size), heads, out_channels]
            x_j: Target node features of shape [(topk x N x batch_size), heads, out_channels]
            edge_index_i: Source node indices of shape [(topk x N x batch_size)]
            size_i: Number of source nodes (N x batch_size)
            embedding: Node embeddings of shape [(N x batch_size), heads, out_channels]
            edges: Edge indices of shape [2, (topk x N x batch_size)]

        Returns:
            Attention-weighted node features.
        """
        # Transform to [(topk x N x batch_size), heads, out_channels]
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        # Get embeddings for source and target nodes
        # [(topk x N x batch_size), self.heads, out_channels]
        embedding_i = embedding[edge_index_i].unsqueeze(1).repeat(1, self.heads, 1)
        embedding_j = embedding[edges[0]].unsqueeze(1).repeat(1, self.heads, 1)

        # Compute cosine similarity [(topk x N x batch_size), heads, 1]
        alpha = torch.sum(embedding_i * embedding_j, dim=-1, keepdim=True)
        # Apply softmax normalization
        alpha = softmax(alpha, edge_index_i, None, size_i)

        # Save attention weights
        self.alpha = alpha

        # Return attention-weighted node features
        return x_j * alpha

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"heads={self.heads}, "
            f"negative_slope={self.negative_slope}, "
            f"dropout={self.dropout})"
        )
