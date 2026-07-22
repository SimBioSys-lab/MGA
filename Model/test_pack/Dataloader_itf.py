import torch
from torch.utils.data import Dataset
import numpy as np

class SequenceParatopeDataset(Dataset):
    """
    Dataset class where MSA sequences are the input, and paratope, secondary structure (ss),
    SASA (sasa), SAPT, and edge data are the targets.
    """
    def __init__(self, data_file, sequence_file, edge_file, max_len=7000):
        """
        Args:
            data_file (str): Path to the .npz file containing ss, sasa, pt, and sapt data.
            sequence_file (str): Path to the .npz file containing sequence data.
            edge_file (str): Path to the .npz file containing edge data.
            max_len (int): Maximum sequence length for padding/truncation.
        """
        # Load sequence, data, and edge files
        self.sequence_data = np.load(sequence_file, allow_pickle=True)
        self.pt_data = np.load(data_file, allow_pickle=True)
        self.edge_data = np.load(edge_file, allow_pickle=True)
        self.max_len = max_len

        # Use keys for indexing
        self.keys = list(self.sequence_data.keys())
        if not (set(self.keys) == set(self.pt_data.keys()) == set(self.edge_data.keys())):
            raise ValueError("Keys in sequence, data, and edge files do not match.")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # Retrieve the key for this index
        key = self.keys[idx]

        # Load and pad/truncate sequence
        sequence = self.sequence_data[key]
        if sequence.shape[1] < self.max_len:
            sequence = np.pad(sequence, ((0, 0), (0, self.max_len - sequence.shape[1])), constant_values=-1)
        sequence = sequence[:, :self.max_len]

        if sequence.shape[0] < 64:
            sequence = np.pad(sequence, ((0, 64 - sequence.shape[0]), (0, 0)), constant_values=-1)
        sequence = sequence[:64, :]

        # Load and pad/truncate pt
        pt = self.pt_data[key]
        pt = (list(pt) + [-1] * self.max_len)[:self.max_len]

        # Load edges
        edges = self.edge_data[key]

        # Convert to PyTorch tensors
        sequence_tensor = torch.tensor(sequence, dtype=torch.long)
        pt_tensor = torch.tensor(pt, dtype=torch.long)
        edges_tensor = torch.tensor(edges, dtype=torch.long)

        # Return data as a tuple
        return sequence_tensor, pt_tensor, edges_tensor

