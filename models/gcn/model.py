import torch
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
            # Create batch-wise graph connectivity by repeating edge_index
            batch_edge_index = edge_index.repeat(1, batch_size)
            offset = (
                torch.arange(batch_size, device=edge_index.device).repeat_interleave(
                    edge_index.size(1)
                )
                * num_nodes
            )
            batch_edge_index = batch_edge_index + offset.view(1, -1)

            h = conv(h, batch_edge_index)
            h = self.tanh(h)

        out = self.regressor(h)
        out = out.reshape(batch_size, 1, num_nodes)

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
