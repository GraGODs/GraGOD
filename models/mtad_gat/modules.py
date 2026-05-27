import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    1-D Convolution layer to extract high-level features of each time-series input.
    Args:
        n_features: Number of input features/nodes
        window_size: length of the input sequence
        kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features: int, kernel_size: int = 7):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(
            in_channels=n_features, out_channels=n_features, kernel_size=kernel_size
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the ConvLayer
        Args:
            x: input tensor of shape (b, n, k):
                b - batch size, n - window size, k - number of features
        returns:
            output tensor of shape (b, n, k)
        """
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back


class FeatureAttentionLayer(nn.Module):
    """
    Single Graph Feature/Spatial Attention Layer
    Args:
        n_features: Number of input features/nodes
        window_size: length of the input sequence
        dropout: percentage of nodes to dropout
        alpha: negative slope used in the leaky rely activation function
        embed_dim: embedding dimension (output dimension of linear transformation)
        use_gatv2: whether to use the modified attention mechanism of GATv2 instead of
            standard GAT
        use_bias: whether to include a bias term in the attention layer
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        dropout: float,
        alpha: float,
        embed_dim: int | None = None,
        use_gatv2: bool = True,
        use_bias: bool = True,
    ):
        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        self.use_gatv2 = use_gatv2
        self.num_nodes = n_features
        self.use_bias = use_bias

        # Because linear transformation is done after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * window_size
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = window_size
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(n_features, n_features))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the FeatureAttentionLayer.
        Args:
            x: input tensor of shape (b, n, k):
                b - batch size, n - window size, k - number of features
        returns:
            TODO: Check this
        """
        # For feature attention we represent a node as the values of a particular
        # feature across all timestamps

        x = x.permute(0, 2, 1)

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied
        # after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)  # (b, k, k, 2*window_size)
            a_input = self.leakyrelu(self.lin(a_input))  # (b, k, k, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)  # (b, k, k, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)  # (b, k, k, embed_dim)
            a_input = self._make_attention_input(Wx)  # (b, k, k, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)  # (b, k, k, 1)

        if self.use_bias:
            e += self.bias

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)
        # Computing new node features using the attention
        h = self.sigmoid(torch.matmul(attention, x))

        return h.permute(0, 2, 1)

    def _make_attention_input(self, v: torch.Tensor):
        """
        Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        Each node consists of all values of that node within the window
            v1 || v1,
            ...
            v1 || vK,
            v2 || v1,
            ...
            v2 || vK,
            ...
            ...
            vK || v1,
            ...
            vK || vK,
        Args:
            v: features tensor of shape (b, k, n):
                b - batch size, k - number of features, n - window size
            returns:
                # TODO: Check this
        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat(
            (blocks_repeating, blocks_alternating), dim=2
        )  # (b, K*K, 2*window_size)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.window_size)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class TemporalAttentionLayer(nn.Module):
    """
    Single Graph Temporal Attention Layer
    Args:
        n_features: number of input features/nodes
        window_size: length of the input sequence
        dropout: percentage of nodes to dropout
        alpha: negative slope used in the leaky rely activation function
        embed_dim: embedding dimension (output dimension of linear transformation)
        use_gatv2: whether to use the modified attention mechanism of GATv2 instead of
            standard GAT
        use_bias: whether to include a bias term in the attention layer
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        dropout: float,
        alpha: float,
        embed_dim: int | None = None,
        use_gatv2: bool = True,
        use_bias: bool = True,
    ):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias

        # Because linear transformation is performed after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * n_features
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = n_features
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(window_size, window_size))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input tensor of shape (b, n, k):
                b - batch size, n - window size, k - number of features
        return:
            TODO: Check this

        """
        # For temporal attention a node is represented as all feature values at a
        # specific timestamp

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied
        # after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)  # (b, n, n, 2*n_features)
            a_input = self.leakyrelu(self.lin(a_input))  # (b, n, n, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)  # (b, n, n, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)  # (b, n, n, embed_dim)
            a_input = self._make_attention_input(Wx)  # (b, n, n, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)  # (b, n, n, 1)

        if self.use_bias:
            e += self.bias  # (b, n, n, 1)

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        h = self.sigmoid(torch.matmul(attention, x))  # (b, n, k)

        return h

    def _make_attention_input(self, v: torch.Tensor):
        """
        Preparing the temporal attention mechanism.
        Creating matrix with all possible combinations of concatenations of node values:
            (v1, v2..)_t1 || (v1, v2..)_t1
            (v1, v2..)_t1 || (v1, v2..)_t2

            ...
            ...

            (v1, v2..)_tn || (v1, v2..)_t1
            (v1, v2..)_tn || (v1, v2..)_t2

        Args:
            v: features tensor of shape (b, n, k):
                b - batch size, n - window size, k - number of features
        returns:
            attention input tensor of shape (b, n, n, 2*k)
        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.n_features)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class GRULayer(nn.Module):
    """
    Gated Recurrent Unit (GRU) Layer
    Args:
        in_dim: number of input features
        hid_dim: hidden size of the GRU
        n_layers: number of layers in GRU
        dropout: dropout rate
    """

    def __init__(self, in_dim: int, hid_dim: int, n_layers: int, dropout: float):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(
            in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the GRU layer
        Args:
            x: input tensor of shape (b, n, k):
                b - batch size, n - window size, k - number of features
        returns:
            # TODO: Check this
        """
        out, h = self.gru(x)
        out, h = out[-1, :, :], h[-1, :, :]  # Extracting from last layer
        return out, h


class RNNDecoder(nn.Module):
    """
    GRU-based Decoder network that converts latent vector into output
    Args:
        in_dim: number of input features
        n_layers: number of layers in RNN
        hid_dim: hidden size of the RNN
        dropout: dropout rate
    """

    def __init__(self, in_dim: int, hid_dim: int, n_layers: int, dropout: float):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(
            in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input tensor of shape (b, n, k):
                b - batch size, n - window size, k - number of features
        returns:
            TODO: Check this
        """
        decoder_out, _ = self.rnn(x)
        return decoder_out


class ReconstructionModel(nn.Module):
    """
    Reconstruction Model
    Args:
        window_size: length of the input sequence
        in_dim: number of input features
        n_layers: number of layers in RNN
        hid_dim: hidden size of the RNN
        in_dim: number of output features
        dropout: dropout rate
    """

    def __init__(
        self,
        window_size: int,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        n_layers: int,
        dropout: float,
    ):
        super(ReconstructionModel, self).__init__()
        self.window_size = window_size
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input tensor of shape (b, n, k):
                b - batch size, n - window size, k - number of features
        returns:
            TODO: Check this
        """
        # x will be last hidden state of the GRU layer
        h_end = x
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(
            x.size(0), self.window_size, -1
        )

        decoder_out = self.decoder(h_end_rep)
        out = self.fc(decoder_out)
        return out


class Forecasting_Model(nn.Module):
    """
    Forecasting model (fully-connected network)
    Args:
        in_dim: number of input features
        hid_dim: hidden size of the FC network
        out_dim: number of output features
        n_layers: number of FC layers
        dropout: dropout rate
    """

    def __init__(
        self, in_dim: int, hid_dim: int, out_dim: int, n_layers: int, dropout: float
    ):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input tensor of shape (b, n, k):
                b - batch size, n - window size, k - number of features
        returns:
            TODO: Check this
        """
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)
