import json
import torch
from torch.utils.data import Dataset
import numpy as np

class CellDataset(Dataset):
    """
    A PyTorch Dataset class for handling cell sentences data. This class processes input cell sentences
    and prepares them for downstream tasks.

    Args:
        cell_sentences (list or other): Input cell sentences, where each cell sentence corresponds to
            a list of cCRE indices.
        max_length (int): Maximum length of the cell sentences, including special tokens (default is 8192).
        is_random (bool): Whether to randomly sample indices if a cell sentence exceeds max_length-2 (default is False).

    Raises:
        ValueError: If input cell_sentences is not properly formatted.
    """
    def __init__(self, cell_sentences, max_length=8192, is_random=True):
        self.max_length = max_length
        self.is_random = is_random

        # Ensure cell_sentences is a list
        if not isinstance(cell_sentences, list):
            cell_sentences = list(cell_sentences)

        # Process string elements in the list
        try:
            cell_sentences = [json.loads(instance) if isinstance(instance, str) else instance for instance in cell_sentences]
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError("Error parsing cell_sentences. Ensure all string elements are JSON serializable lists.")

        # Validate that each element in cell_sentences is a list of integers
        if not all(isinstance(instance, list) and all(isinstance(i, int) for i in instance) for instance in cell_sentences):
            raise ValueError("Invalid format for cell_sentences. Each element must be a list of integers.")

        self.cell_sentences = cell_sentences

    def __len__(self):
        """
        Returns the number of cell sentences in the dataset.

        Returns:
            int: Number of cell sentences.
        """
        return len(self.cell_sentences)

    def __getitem__(self, idx):
        """
        Retrieves a single cell sentence and applies necessary preprocessing.

        Args:
            idx (int): Index of the cell sentence to retrieve.

        Returns:
            torch.Tensor: Processed cell sentence as a tensor of integers.
        """
        cell = np.array(self.cell_sentences[idx])

        # Truncate or randomly sample if the cell sentence exceeds max_length-2
        if len(cell) > self.max_length - 2:
            if self.is_random:
                index = np.sort(np.random.choice(range(len(cell)), self.max_length - 2, replace=False))
                cell = cell[index]
            else:
                cell = cell[:self.max_length - 2]

        # Add special tokens (1 for [CLS], 2 for [SEP])
        cell = [1] + cell.tolist() + [2]

        # Convert to torch tensor
        return torch.tensor(cell, dtype=torch.long)

def collate_fn(data):
    """
    Collate function for padding sequences to the same length in a batch.

    This function ensures that all sequences in a batch have the same length by padding shorter sequences
    with the [PAD] token, whose ID in the EpiAgent model is 0.

    Args:
        data (list of list of int): A list of sequences, where each sequence is a list of token IDs.

    Returns:
        torch.Tensor: A padded tensor of shape [batch_size, max_seq_length], where max_seq_length is the length
        of the longest sequence in the batch.
    """
    max_len = max([len(row) for row in data])
    # Pad all sequences to the maximum length in the batch
    padded_data = [
        torch.nn.functional.pad(
            torch.LongTensor(row), pad=(0, max_len - len(row)), mode='constant', value=0
        ) for row in data
    ]
    # Stack all padded sequences into a single tensor
    stacked_data = torch.stack(padded_data)
    return stacked_data



class CellDatasetForSCA(Dataset):
    """
    A PyTorch Dataset class for handling cell sentences and cell type annotations.
    This class prepares input cell sentences and corresponding cell types for supervised
    cell type annotation tasks with the EpiAgent_supervised model.

    Args:
        cell_sentences (list): A list of cell sentences, where each sentence corresponds
            to a list of cCRE indices.
        cell_types (list): A list of cell type labels corresponding to each cell sentence.
        max_length (int): Maximum length of the cell sentences, including special tokens (default is 8192).
        is_random (bool): Whether to randomly sample indices if a cell sentence exceeds max_length-2 (default is False).

    Raises:
        ValueError: If input cell_sentences is not properly formatted or does not align with cell_types.
    """
    def __init__(self, cell_sentences, cell_types, max_length=8192, is_random=True):
        self.max_length = max_length
        self.is_random = is_random

        # Ensure cell_sentences is a list and parse JSON strings
        try:
            self.cell_sentences = [
                json.loads(instance) if isinstance(instance, str) else instance
                for instance in cell_sentences
            ]
        except (json.JSONDecodeError, TypeError):
            raise ValueError("Error parsing cell_sentences. Ensure all string elements are JSON serializable lists.")

        # Validate format of cell_sentences
        if not all(isinstance(sentence, list) and all(isinstance(i, int) for i in sentence) for sentence in self.cell_sentences):
            raise ValueError("Invalid format for cell_sentences. Each element must be a list of integers.")

        # Validate that cell_sentences and cell_types are of the same length
        if len(self.cell_sentences) != len(cell_types):
            raise ValueError("Mismatch between cell_sentences and cell_types lengths.")

        self.cell_types = cell_types

    def __len__(self):
        """
        Returns the number of cell sentences in the dataset.

        Returns:
            int: Number of cell sentences.
        """
        return len(self.cell_sentences)

    def __getitem__(self, idx):
        """
        Retrieves a single cell sentence and its corresponding cell type.

        Args:
            idx (int): Index of the cell sentence to retrieve.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Processed cell sentence as a tensor of integers.
                - torch.Tensor: Corresponding cell type label as a tensor.
        """
        cell = np.array(self.cell_sentences[idx])
        cell_type = self.cell_types[idx]

        # Truncate or randomly sample if the cell sentence exceeds max_length-2
        if len(cell) > self.max_length - 2:
            if self.is_random:
                index = np.sort(np.random.choice(range(len(cell)), self.max_length - 2, replace=False))
                cell = cell[index]
            else:
                cell = cell[:self.max_length - 2]

        # Add special tokens (1 for [CLS], 2 for [SEP])
        cell = [1] + cell.tolist() + [2]

        # Convert to torch tensors
        return torch.tensor(cell, dtype=torch.long), torch.tensor(cell_type, dtype=torch.long)

def collate_fn_for_SCA(data):
    """
    Collate function for supervised cell type annotation (SCA) tasks.

    This function pads sequences to the same length in a batch and organizes cell sentences
    and cell type labels into separate tensors.

    Args:
        data (list of tuple): A list where each element is a tuple containing:
            - list of int: A cell sentence (sequence of token IDs).
            - int: The corresponding cell type label.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: A padded tensor of shape [batch_size, max_seq_length], where max_seq_length
              is the length of the longest sequence in the batch.
            - torch.Tensor: A tensor of shape [batch_size] containing the cell type labels.
    """
    # Unpack cell sentences and cell type labels
    cells, cell_types = zip(*data)

    # Determine the maximum length of sequences in the batch
    max_len = max(len(row) for row in cells)

    # Pad all sequences to the maximum length
    padded_cells = [
        torch.nn.functional.pad(
            torch.LongTensor(row), pad=(0, max_len - len(row)), mode='constant', value=0
        ) for row in cells
    ]

    # Stack padded sequences and convert cell type labels to a tensor
    stacked_cells = torch.stack(padded_cells)
    cell_types_tensor = torch.tensor(cell_types, dtype=torch.long)

    return stacked_cells, cell_types_tensor