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
        mlp (nn.ModuleList): A list of modules representing the layers of the MLP.

    Args:
        in_num (int): The number of input features.
        layer_num (int): The number of layers in the MLP.
        inter_num (int, optional): The number of neurons in intermediate layers. Defaults to 512.
    """

    def __init__(self, in_num, layer_num, inter_num=512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num - 1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in_num, inter_num))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        """
        Forward pass of the OutLayer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the MLP.
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

    This layer applies a graph convolution followed by batch normalization and ReLU activation.

    Attributes:
        gnn (GraphLayer): The graph convolutional layer.
        bn (nn.BatchNorm1d): Batch normalization layer.
        relu (nn.ReLU): ReLU activation function.

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        heads (int, optional): Number of attention heads. Defaults to 1.
    """

    def __init__(self, in_channel, out_channel, heads=1):
        super(GNNLayer, self).__init__()

        self.gnn = GraphLayer(in_channel, out_channel, heads=heads, concat_heads=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):
        """
        Forward pass of the GNNLayer.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Graph connectivity in COO format.
            embedding (torch.Tensor, optional): Node embeddings. Defaults to None.
            node_num (int, optional): Number of nodes. Defaults to 0.

        Returns:
            torch.Tensor: Output tensor after applying graph convolution, batch normalization, and ReLU.
        """
        out = self.gnn(x, edge_index, embedding)
        out = self.bn(out)
        return self.relu(out)


class GraphLayer(MessagePassing):
    """
    Class for graph convolutional layers using message passing.

    Attributes:
        in_channels (int): Number of input channels for the layer
        out_channels (int): Number of output channels for the layer
        heads (int): Number of heads for multi-head attention
        concat_heads (bool): Whether to concatenate across heads
        negative_slope (float): Slope for LeakyReLU
        dropout (float): Dropout rate
        lin (nn.Module): Linear layer for transforming input
        att_i (nn.Parameter): Attention parameter related to x_i
        att_j (nn.Parameter): Attention parameter related to x_j
        att_em_i (nn.Parameter): Attention parameter related to embedding of x_i
        att_em_j (nn.Parameter): Attention parameter related to embedding of x_j
        bias (nn.Parameter): Bias parameter added after message propagation

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        heads (int, optional): Number of attention heads. Defaults to 1.
        concat_heads (bool, optional): Whether to concatenate attention heads. Defaults to True.
        negative_slope (float, optional): Negative slope for LeakyReLU. Defaults to 0.2.
        dropout (float, optional): Dropout rate. Defaults to 0.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat_heads=True,
        negative_slope=0.2,
        dropout=0,
    ):
        super(GraphLayer, self).__init__(aggr="add", node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat_heads = concat_heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        # parameters related to weight matrix W
        self.lin = Linear(in_channels, heads * out_channels, bias=False)

        # attention parameters related to x_i, x_j
        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))

        # attention parameters related embeddings v_i, v_j
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))

        # if concatenating the across heads, consider the change of out_channels
        self._out_channels = heads * out_channels if concat_heads else out_channels
        self.bias = Parameter(torch.Tensor(self._out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialise parameters of GraphLayer."""
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        zeros(self.att_em_i)
        zeros(self.att_em_j)
        self.bias.data.zero_()

    def forward(self, x, edge_index, embedding):
        """Forward method for propagating messages of GraphLayer.

        Args:
            x (torch.Tensor): Node features tensor of shape [N x batch_size, in_channels],
                              where N is the number of nodes.
            edge_index (torch.Tensor): Graph connectivity in COO format,
                                       shape [2, E x batch_size], where E is the number of edges.
            embedding (torch.Tensor): Node embeddings tensor of shape [N x batch_size, out_channels].

        Returns:
            torch.Tensor: Output tensor after message passing and attention mechanism.
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

        # transform [N x batch_size, 1, _out_channels] to [N x batch_size, _out_channels]
        out = out.view(-1, self._out_channels)

        # apply final bias vector
        out += self.bias

        return out

    def message(self, x_i, x_j, edge_index_i, size_i, embedding, edges):
        """Calculate the attention weights using the embedding vector, eq (6)-(8) in [1].

        Args:
            x_i (torch.Tensor): Source node features of shape [(topk x N x batch_size), out_channels].
            x_j (torch.Tensor): Target node features of shape [(topk x N x batch_size), out_channels].
            edge_index_i (torch.Tensor): Source node indices of shape [(topk x N x batch_size)].
            size_i (int): Number of source nodes (N x batch_size).
            embedding (torch.Tensor): Node embeddings of shape [(N x batch_size), out_channels].
            edges (torch.Tensor): Edge indices of shape [2, (topk x N x batch_size)].

        Returns:
            torch.Tensor: Attention-weighted node features.
        """
        # transform to [(topk x N x batch_size), 1, out_channels]
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        if embedding is not None:
            # [(topk x N x batch_size), 1, out_channels]
            embedding_i = embedding[edge_index_i].unsqueeze(1).repeat(1, self.heads, 1)
            embedding_j = embedding[edges[0]].unsqueeze(1).repeat(1, self.heads, 1)

            # [(topk x N x batch_size), 1, 2 x out_channels]
            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat(
                (x_j, embedding_j), dim=-1
            )  # concatenates along the last dim, i.e. columns in this case

        # concatenate learnable parameters to become [1, 1, 2 x out_channels]
        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)

        # [(topk x N x batch_size), 1]
        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(
            -1
        )  # the matrix multiplication between a^T and g's in eqn (7)

        alpha = alpha.view(-1, self.heads, 1)  # [(topk x N x batch_size), 1, 1]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, None, size_i)  # eqn (8)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # save to self
        self.alpha = alpha

        # multiply node feature by alpha
        return x_j * alpha

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads})"
