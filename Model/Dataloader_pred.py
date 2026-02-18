import torch
from torch.utils.data import Dataset
import numpy as np


class SequenceParatopeDataset(Dataset):
    """
    Prediction-only dataset.
    Returns:
        sequence_tensor, edges_tensor
    No labels (pt / ss / sasa / etc).
    """

    def __init__(self, sequence_file, edge_file, max_len=7000):
        """
        Args:
            sequence_file (str): Path to the .npz file containing sequence data.
            edge_file (str): Path to the .npz file containing edge data.
            max_len (int): Maximum sequence length for padding/truncation.
        """
        self.sequence_data = np.load(sequence_file, allow_pickle=True)
        self.edge_data = np.load(edge_file, allow_pickle=True)
        self.max_len = max_len

        self.keys = list(self.sequence_data.keys())

        # Ensure keys match between sequence and edge files
        if set(self.keys) != set(self.edge_data.keys()):
            missing = set(self.keys) ^ set(self.edge_data.keys())
            raise ValueError(
                f"Keys in sequence and edge files do not match. Mismatches: {missing}"
            )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        """
        Returns:
            sequence_tensor (Torch long) : shape (64, L)
            edges_tensor    (Torch long) : shape (E, 2)
        """
        key = self.keys[idx]

        # --- Sequence ---
        sequence = self.sequence_data[key]   # shape (depth, length)

        # pad / truncate width
        if sequence.shape[1] < self.max_len:
            sequence = np.pad(
                sequence,
                ((0, 0), (0, self.max_len - sequence.shape[1])),
                constant_values=-1
            )
        sequence = sequence[:, :self.max_len]

        # pad / truncate depth (MSA depth = 64)
        if sequence.shape[0] < 64:
            sequence = np.pad(
                sequence,
                ((0, 64 - sequence.shape[0]), (0, 0)),
                constant_values=-1
            )
        sequence = sequence[:64, :]

        # --- Edges ---
        edges = self.edge_data[key]   # shape (E, 2)

        # Convert to tensors
        sequence_tensor = torch.tensor(sequence, dtype=torch.long)
        edges_tensor = torch.tensor(edges, dtype=torch.long)

        return sequence_tensor, edges_tensor, key

