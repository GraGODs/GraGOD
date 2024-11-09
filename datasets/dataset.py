import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    """
    A PyTorch Dataset that creates sliding windows over time series data.

    It creates windows of fixed size that slide over the input data,
    optionally handling labels and graph edge indices.

    Args:
        data (torch.Tensor): The input time series data
        window_size (int): The size of each sliding window
        edge_index (torch.Tensor, optional): Edge indices defining graph connectivityg
        labels (torch.Tensor, optional): Labels for each timestep
        horizon (int, optional): Number of future timesteps to predict.

    Attributes:
        data (torch.Tensor): The input time series data
        labels (torch.Tensor): Labels for each timestep
        edge_index (torch.Tensor): Edge indices defining graph connectivity
        window_size (int): The size of each sliding window
        horizon (int): Number of future timesteps to predict
        valid_indices (torch.Tensor): Indices of valid windows
    """

    def __init__(
        self,
        data: torch.Tensor,
        window_size: int,
        edge_index: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        horizon: int = 1,
    ):
        self.data = data
        self.labels = labels
        self.edge_index = edge_index
        self.window_size = window_size
        self.horizon = horizon

        self.valid_indices = self._get_valid_indices()

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        valid_idx = self.valid_indices[index]

        x = self.data[valid_idx : valid_idx + self.window_size]
        y = self.data[
            valid_idx + self.window_size : valid_idx + self.window_size + self.horizon
        ]

        out_labels = (
            self.labels[
                valid_idx
                + self.window_size : valid_idx
                + self.window_size
                + self.horizon
            ]
            if self.labels is not None
            else None
        )

        return x, y, out_labels, self.edge_index

    def _get_valid_indices(self):
        """
        Determines valid starting indices for sliding windows.

        Returns:
            torch.Tensor: Array of valid starting indices for sliding windows
        """
        total_windows = len(self.data) - self.window_size

        if self.labels is None:
            # if there are no labels, all indices are valid
            print(f"No labels provided - using all {total_windows} windows")
            return torch.arange(total_windows)

        # an index is valid if all the labels between
        # [index, index + window_size + horizon] are 0
        valid_indices_mask = [
            not torch.any(
                self.labels[i : i + self.window_size + self.horizon],
            )
            for i in range(len(self.data) - self.window_size - self.horizon)
        ]

        valid_indices = torch.where(torch.tensor(valid_indices_mask))[0]
        print(
            f"Found {len(valid_indices)} valid windows"
            f" out of {total_windows} total windows"
        )

        return valid_indices
