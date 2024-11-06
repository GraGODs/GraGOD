import numpy as np
import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray | torch.Tensor,
        window_size: int,
        horizon: int = 1,
    ):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon

    def __getitem__(self, index):
        x = self.data[index : index + self.window_size]
        y = self.data[
            index + self.window_size : index + self.window_size + self.horizon
        ]
        return x, y

    def __len__(self):
        return len(self.data) - self.window_size
