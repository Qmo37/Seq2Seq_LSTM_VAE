"""
PyTorch Dataset class for learning behavior sequences.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class LearningBehaviorDataset(Dataset):
    """
    PyTorch Dataset for learning behavior sequences.

    Args:
        X: Input sequences of shape (n_samples, seq_len, n_features)
        y: Output sequences of shape (n_samples, out_len, 1)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx]
