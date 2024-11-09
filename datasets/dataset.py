import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
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
        if index >= len(self.valid_indices):
            raise IndexError("Index out of range")

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
        if self.labels is None:
            # if there are no labels, all indices are valid
            return torch.arange(len(self.data) - self.window_size)

        # an index is valid if all the labels between
        # [index, index + window_size + horizon] are 0
        valid_indices_mask = [
            not torch.any(
                self.labels[i : i + self.window_size + self.horizon],
            )
            for i in range(len(self.data) - self.window_size - self.horizon)
        ]

        valid_indices = torch.where(torch.tensor(valid_indices_mask))[0]

        return valid_indices
