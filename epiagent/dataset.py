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

class CellDatasetForUFE(Dataset):
    """
    A PyTorch Dataset class for Unsupervised Feature Extraction (UFE) fine-tuning. This class processes input cell sentences
    and prepares them along with associated signals and data required for the Cell-cCRE Alignment (CCA) task.

    Args:
        adata (AnnData): Annotated data matrix containing the signal information for each cell.
        cell_sentences (list of list of int): List of cell sentences, where each cell sentence is a list of cCRE indices.
        max_length (int): Maximum length of the cell sentences, including special tokens [CLS] and [SEP] (default is 8192).
        alpha_for_CCA (int): Multiplier to determine the number of inaccessible cCREs sampled for the CCA task (default is 5).
        num_cCRE (int): Total number of candidate cis-regulatory elements (cCREs) (default is 1,355,445).
        is_random (bool): Whether to randomly sample indices if a cell sentence exceeds `max_length - 2` (default is False).

    Raises:
        ValueError: If `cell_sentences` is not properly formatted.
    """
    def __init__(
        self,
        adata,
        cell_sentences,
        max_length=8192,
        alpha_for_CCA=1,
        num_cCRE=1355445,
        is_random=False
    ):
        self.adata = adata.copy()
        self.adata.X[self.adata.X>0] = 1.
        self.max_length = max_length
        self.alpha_for_CCA = alpha_for_CCA
        self.is_random = is_random
        self.num_cCRE = num_cCRE

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
        Retrieves and processes a single cell sentence along with additional data required for UFE fine-tuning.

        Args:
            idx (int): Index of the cell sentence to retrieve.

        Returns:
            tuple:
                - cell_tensor (torch.Tensor): Processed cell sentence as a tensor of cCRE indices with special tokens.
                - signals_tensor (torch.Tensor): Signal values corresponding to the cCREs in the cell.
                - ex_cell_ccre_ids_for_CCA_tensor (torch.Tensor): Tensor of cCRE indices used in the CCA task.
                - ex_cell_ccre_accessibility_for_CCA_tensor (torch.Tensor): Tensor indicating accessibility labels for cCREs in the CCA task.
        """
        # Retrieve the cell sentence (list of cCRE indices)
        cell = np.array(self.cell_sentences[idx], dtype=int)

        # Retrieve the signal data for the cell from `adata`
        signals = self.adata[idx].X.toarray().reshape(-1)

        # ---------------------------------------------
        # Prepare data for the Cell-cCRE Alignment (CCA) task
        # ---------------------------------------------

        # Create a boolean array indicating all cCREs as inaccessible initially
        inaccessible_indices = np.ones(self.num_cCRE, dtype=bool)

        # Adjust cCRE indices if they start from an offset (assuming 4 as per the original code)
        adjusted_cell_indices = cell - 4  # Adjusting indices if necessary

        # Mark the accessible cCREs (those present in the cell) as False (i.e., not inaccessible)
        inaccessible_indices[adjusted_cell_indices] = False

        # Generate the list of inaccessible cCRE indices
        cell_inaccessible_indices = np.where(inaccessible_indices)[0] + 4  # Re-adjust indices if needed

        # Determine the number of inaccessible cCREs to sample based on `alpha_for_CCA`
        num_inaccessible_to_sample = int(len(cell) * self.alpha_for_CCA)

         # Randomly sample inaccessible cCREs without replacement
        if num_inaccessible_to_sample <= len(cell_inaccessible_indices):
            sampled_inaccessible_ccre = np.random.choice(
                cell_inaccessible_indices,
                size=num_inaccessible_to_sample,
                replace=False
            )
        else: 
            sampled_inaccessible_ccre = np.random.choice(
                cell_inaccessible_indices,
                size=num_inaccessible_to_sample,
                replace=True
            )

        # Combine accessible and sampled inaccessible cCRE indices for the CCA task
        ex_cell_ccre_ids_for_CCA = np.concatenate([cell, sampled_inaccessible_ccre])

        # Create accessibility labels: 1 for accessible cCREs, 0 for inaccessible cCREs
        ex_cell_ccre_accessibility_for_CCA = np.concatenate([
            np.ones_like(cell),
            np.zeros_like(sampled_inaccessible_ccre)
        ])

        # ---------------------------------------------
        # Truncate or sample the cell sentence if it exceeds `max_length - 2`
        # ---------------------------------------------

        # Check if the cell sentence needs to be truncated or sampled
        if len(cell) > self.max_length - 2:
            if self.is_random:
                # Randomly sample indices to reduce the length to `max_length - 2`
                sampled_indices = np.sort(
                    np.random.choice(
                        range(len(cell)),
                        size=self.max_length - 2,
                        replace=False
                    )
                )
                cell = cell[sampled_indices]
            else:
                # Truncate the cell sentence to the first `max_length - 2` elements
                cell = cell[:self.max_length - 2]

        # Add special tokens: [CLS] token ID is 1 at the beginning, [SEP] token ID is 2 at the end
        cell_with_special_tokens = [1] + cell.tolist() + [2]

        # ---------------------------------------------
        # Convert data to PyTorch tensors
        # ---------------------------------------------

        cell_tensor = torch.tensor(cell_with_special_tokens, dtype=torch.long)
        signals_tensor = torch.tensor(signals, dtype=torch.float32)
        ex_cell_ccre_ids_for_CCA_tensor = torch.tensor(ex_cell_ccre_ids_for_CCA, dtype=torch.long)
        ex_cell_ccre_accessibility_for_CCA_tensor = torch.tensor(ex_cell_ccre_accessibility_for_CCA, dtype=torch.float32)

        return (
            cell_tensor,
            signals_tensor,
            ex_cell_ccre_ids_for_CCA_tensor,
            ex_cell_ccre_accessibility_for_CCA_tensor
        )


def collate_fn_for_UFE(data):
    """
    Collate function for padding sequences and preparing batches for UFE fine-tuning.

    This function pads the cell sequences to the same length within a batch and
    compiles the necessary data for the Cell-cCRE Alignment (CCA) task.

    Args:
        data (list of tuples): Each tuple contains:
            - cell_tensor (torch.Tensor): Processed cell sequence tensor.
            - signals_tensor (torch.Tensor): Signal values tensor.
            - ex_cell_ccre_ids_for_CCA_tensor (torch.Tensor): cCRE indices tensor for the CCA task.
            - ex_cell_ccre_accessibility_for_CCA_tensor (torch.Tensor): Accessibility labels tensor for the CCA task.

    Returns:
        tuple:
            - padded_cells_batch (torch.Tensor): Padded batch of cell sequences.
            - signals_batch (torch.Tensor): Batch of signal arrays.
            - ex_cell_ccre_ids_batch (list of torch.Tensor): List of cCRE indices tensors for the CCA task for each sample.
            - ex_cell_ccre_accessibility_batch (torch.Tensor): Concatenated accessibility labels tensor for the CCA task.
    """
    # Unzip the data into individual components
    cells_list, signals_list, ex_cell_ccre_ids_list, ex_cell_ccre_accessibility_list = zip(*data)

    # Determine the maximum sequence length in the batch
    max_seq_length = max([cell.size(0) for cell in cells_list])

    # Pad the cell sequences to the maximum length using the PAD token ID (0)
    padded_cells_batch = torch.stack([
        torch.nn.functional.pad(
            cell,
            pad=(0, max_seq_length - cell.size(0)),
            mode='constant',
            value=0  # PAD token ID is 0
        ) for cell in cells_list
    ])

    # Stack the signals into a batch tensor
    signals_batch = torch.stack(signals_list)

    # Collect cCRE IDs for the CCA task into a list (they can have variable lengths)
    ex_cell_ccre_ids_batch = list(ex_cell_ccre_ids_list)

    # Concatenate accessibility labels for the CCA task into a single tensor
    ex_cell_ccre_accessibility_batch = torch.cat(ex_cell_ccre_accessibility_list)

    return (
        padded_cells_batch,
        signals_batch,
        ex_cell_ccre_ids_batch,
        ex_cell_ccre_accessibility_batch
    )

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



class CellDatasetForOSP(Dataset):
    """
    A PyTorch Dataset class for handling cell sentences and related data in the task of cellular response prediction of Out-of-Sample Stimulated Perturbation (OSP).
    This class processes input cell sentences, along with predicted cell sentences, condition data, and other related information,
    and prepares them for downstream tasks, specifically for predicting cell states under perturbations.

    Args:
        adata (AnnData): Annotated data matrix containing the signal information for each cell.
        cell_sentences (list of list of int): List of cell sentences, where each cell sentence is a list of cCRE indices.
        predicted_cell_sentences (list of list of int): List of predicted cell sentences, where each predicted sentence corresponds to a list of cCRE indices.
        condition (list of str): List of conditions corresponding to each cell (e.g., "Resting_mapping_to_stimulated").
        max_length (int): Maximum length of the cell sentences, including special tokens [CLS] and [SEP] (default is 8192).
        alpha_for_CCA (int): Multiplier to determine the number of inaccessible cCREs sampled for the CCA task (default is 1).
        num_cCRE (int): Total number of candidate cis-regulatory elements (cCREs) (default is 1,355,445).
        is_random (bool): Whether to randomly sample indices if a cell sentence exceeds max_length-2 (default is False).
        label_for_classification (list): List of labels for classification corresponding to each cell.
        perturbation_labels (list of str): List of strings indicating perturbation states, default is ['resting', 'stimulated'].

    Raises:
        ValueError: If cell_sentences, predicted_cell_sentences, or other arguments are not properly formatted.
    """
    def __init__(self, 
                 adata, 
                 cell_sentences, 
                 predicted_cell_sentences, 
                 condition, 
                 label_for_classification, 
                 perturbation_labels=['resting', 'stimulated'], 
                 alpha_for_CCA=1, 
                 max_length=8192, 
                 num_cCRE=1355445, 
                 is_random=True):

        self.condition = condition
        self.max_length = max_length
        self.is_random = is_random
        self.alpha_for_CCA = alpha_for_CCA
        self.num_cCRE = num_cCRE
        self.adata = adata
        self.label_for_classification = label_for_classification
        # Validate that perturbation_labels contains exactly two entries
        if len(perturbation_labels) != 2:
            raise ValueError("perturbation_labels must contain exactly two strings, e.g., ['resting', 'stimulated'].")
        
        # Generate the mapping token for the perturbation
        self.mapping_token = f"{perturbation_labels[0]}_mapping_to_{perturbation_labels[1]}"
        
        # Validate the condition column to ensure it has only the expected three values
        if not all(val in [perturbation_labels[0], perturbation_labels[1], self.mapping_token] for val in condition):
            raise ValueError(f"condition must contain only {perturbation_labels[0]}, {perturbation_labels[1]}, and {self.mapping_token}.")
        
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

        # Ensure predicted_cell_sentences is a list and parse JSON strings
        try:
            self.predicted_cell_sentences = [
                json.loads(instance) if isinstance(instance, str) else instance
                for instance in predicted_cell_sentences
            ]
        except (json.JSONDecodeError, TypeError):
            raise ValueError("Error parsing cell_sentences. Ensure all string elements are JSON serializable lists.")

        # Validate format of predicted_cell_sentences
        if not all(isinstance(sentence, list) and all(isinstance(i, int) for i in sentence) for sentence in self.predicted_cell_sentences):
            raise ValueError("Invalid format for cell_sentences. Each element must be a list of integers.")



    def __len__(self):
        """
        Returns the number of cell sentences in the dataset.

        Returns:
            int: Number of cell sentences.
        """
        return len(self.cell_sentences)

    def __getitem__(self, idx):
        """
        Retrieves and processes a single cell sentence along with additional data required for OSP prediction.

        Args:
            idx (int): Index of the cell sentence to retrieve.

        Returns:
            tuple:
                - cell_tensor (torch.Tensor): Processed cell sentence as a tensor of cCRE indices with special tokens.
                - signals_tensor (torch.Tensor): Signal values corresponding to the cCREs in the cell.
                - classification_labels_tensor (torch.Tensor): Tensor for classification labels.
                - ex_cell_ccre_ids_for_CCA_tensor (torch.Tensor): Tensor for the cCRE indices used in the Cell-cCRE Alignment (CCA) task.
                - ex_cell_ccre_accessibility_for_CCA_tensor (torch.Tensor): Tensor for the accessibility labels for cCREs.
        """
        # Retrieve the cell sentence (list of cCRE indices)
        cell = np.array(self.cell_sentences[idx], dtype=int)
        predicted_cell = np.array(self.predicted_cell_sentences[idx], dtype=int)

        # Retrieve the signal data for the cell from `adata`
        signals = self.adata[idx].X.toarray().reshape(-1)

        # ---------------------------------------------
        # Prepare data for the Cell-cCRE Alignment (CCA) task
        # ---------------------------------------------

        # Create a boolean array indicating all cCREs as inaccessible initially
        predicted_inaccessible_indices = np.ones(self.num_cCRE, dtype=bool)

        # Adjust cCRE indices if they start from an offset (assuming 4 as per the original code)
        adjusted_cell_indices = cell - 4  # Adjusting indices if necessary
        predicted_adjusted_cell_indices = predicted_cell - 4

        # Mark the accessible cCREs (those present in the cell) as False (i.e., not inaccessible)
        predicted_inaccessible_indices[predicted_adjusted_cell_indices] = False

        # Generate the list of inaccessible cCRE indices
        predicted_cell_inaccessible_indices = np.where(predicted_inaccessible_indices)[0] + 4  # Re-adjust indices if needed

        # Determine the number of inaccessible cCREs to sample based on `alpha_for_CCA`
        num_inaccessible_to_sample = int(len(predicted_cell) * self.alpha_for_CCA)

        # Randomly sample inaccessible cCREs without replacement
        if num_inaccessible_to_sample <= len(predicted_cell_inaccessible_indices):
            predicted_sampled_inaccessible_ccre = np.random.choice(
                predicted_cell_inaccessible_indices,
                size=num_inaccessible_to_sample,
                replace=False
            )
        else: 
            predicted_sampled_inaccessible_ccre = np.random.choice(
                predicted_cell_inaccessible_indices,
                size=num_inaccessible_to_sample,
                replace=True
            )

        # Combine accessible and sampled inaccessible cCRE indices for the CCA task
        ex_cell_ccre_ids_for_CCA = np.concatenate([predicted_cell, predicted_sampled_inaccessible_ccre])

        # Create accessibility labels: 1 for accessible cCREs, 0 for inaccessible cCREs
        ex_cell_ccre_accessibility_for_CCA = np.concatenate([
            np.ones_like(predicted_cell),
            np.zeros_like(predicted_sampled_inaccessible_ccre)
        ])

        # ---------------------------------------------
        # Truncate or sample the cell sentence if it exceeds `max_length - 2`
        # ---------------------------------------------

        # Check if the cell sentence needs to be truncated or sampled
        if len(cell) > self.max_length - 2:
            if self.is_random:
                # Randomly sample indices to reduce the length to `max_length - 2`
                sampled_indices = np.sort(
                    np.random.choice(
                        range(len(cell)),
                        size=self.max_length - 2,
                        replace=False
                    )
                )
                cell = cell[sampled_indices]
            else:
                # Truncate the cell sentence to the first `max_length - 2` elements
                cell = cell[:self.max_length - 2]

        # Add special tokens: [CLS] token ID is 1 at the beginning, [SEP] token ID is 2 at the end
        if self.condition[idx] == self.mapping_token:
            cell = [self.num_cCRE + 4] + cell.tolist() + [2]  # The special token for perturbation is num_cCRE + 4
        else:
            cell = [1] + cell.tolist() + [2]

        # Convert to torch tensors
        cell_tensor = torch.tensor(cell, dtype=torch.long)
        signals_tensor = torch.tensor(signals, dtype=torch.float32)
        ex_cell_ccre_ids_for_CCA_tensor = torch.tensor(ex_cell_ccre_ids_for_CCA, dtype=torch.long)
        ex_cell_ccre_accessibility_for_CCA_tensor = torch.tensor(ex_cell_ccre_accessibility_for_CCA, dtype=torch.float32)
        classification_labels_tensor = torch.tensor(self.label_for_classification[idx], dtype=torch.long)

        return (
            cell_tensor,
            signals_tensor,
            classification_labels_tensor,
            ex_cell_ccre_ids_for_CCA_tensor,
            ex_cell_ccre_accessibility_for_CCA_tensor
        )


def collate_fn_for_OSP(data):
    """
    Collate function for padding sequences and preparing batches for the OSP task.

    This function pads the cell sequences to the same length within a batch and
    compiles the necessary data for the Cell-cCRE Alignment (CCA) task.

    Args:
        data (list of tuples): Each tuple contains:
            - torch.Tensor: Processed cell sentence tensor.
            - torch.Tensor: Signal values tensor.
            - torch.Tensor: Classification label tensor.
            - torch.Tensor: Tensor for the cCRE indices used in the CCA task.
            - torch.Tensor: Tensor for the accessibility labels of the cCREs.

    Returns:
        tuple: A tuple containing:
            - padded_cells_batch (torch.Tensor): Padded batch of cell sequences.
            - signals_batch (torch.Tensor): Batch of signal arrays.
            - classification_labels_batch (torch.Tensor): Batch of classification labels.
            - ex_cell_ccre_ids_for_CCA_batch (list of torch.Tensor): List of tensors for CCA task.
            - ex_cell_ccre_accessibility_for_CCA_batch (torch.Tensor): Concatenated tensor for CCA task accessibility labels.
    """
    # Unzip the data into individual components
    cells_list, signals_list, classification_labels_list, ex_cell_ccre_ids_for_CCA_list, ex_cell_ccre_accessibility_for_CCA_list = zip(*data)

    # Determine the maximum sequence length in the batch
    max_seq_length = max([cell.size(0) for cell in cells_list])

    # Pad the cell sequences to the maximum length using the PAD token ID (0)
    padded_cells_batch = torch.stack([
        torch.nn.functional.pad(
            cell,
            pad=(0, max_seq_length - cell.size(0)),
            mode='constant',
            value=0  # PAD token ID is 0
        ) for cell in cells_list
    ])

    # Stack the signals and classification labels into batch tensors
    signals_batch = torch.stack(signals_list)
    classification_labels_batch = torch.stack(classification_labels_list)

    # Collect the ex_cell_ccre_ids_for_CCA tensors (for CCA task) into a list
    ex_cell_ccre_ids_for_CCA_batch = list(ex_cell_ccre_ids_for_CCA_list)

    # Concatenate ex_cell_ccre_accessibility_for_CCA into a single tensor
    ex_cell_ccre_accessibility_for_CCA_batch = torch.cat(ex_cell_ccre_accessibility_for_CCA_list)

    return padded_cells_batch, signals_batch, classification_labels_batch, ex_cell_ccre_ids_for_CCA_batch, ex_cell_ccre_accessibility_for_CCA_batch

class TestCellDatasetForOSP(Dataset):
    """
    A PyTorch Dataset class for handling cell sentences and related data during the inference phase of Out-of-Sample Stimulated Perturbation (OSP).
    This class processes input cell sentences and prepares them for downstream tasks, specifically for inferring cellular responses under perturbations.
    
    The main difference between this class and `CellDatasetForOSP` is that it is designed to handle validation/inference datasets and does not involve perturbation-specific mapping.
    """
    def __init__(self, cell_sentences, condition, perturbation_labels=['resting', 'stimulated'], max_length=8192, num_cCRE=1355445, is_random=True):

        self.condition = condition
        self.max_length = max_length
        self.is_random = is_random
        self.num_cCRE = num_cCRE
        
        # Ensure perturbation_labels contains exactly two elements
        if len(perturbation_labels) != 2:
            raise ValueError("perturbation_labels should contain exactly two elements.")
        
        self.new_condition = f"{perturbation_labels[0]}_mapping_to_{perturbation_labels[1]}"
        
        # Validate that condition only contains the specified labels and the new condition
        valid_conditions = perturbation_labels + [self.new_condition]
        if not all(cond in valid_conditions for cond in self.condition):
            raise ValueError(f"Condition values must be one of {valid_conditions}.")
                
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


    def __len__(self):
        """
        Returns the number of cell sentences in the dataset.
        """
        return len(self.cell_sentences)

    def __getitem__(self, idx):
        """
        Retrieves a single cell sentence and applies necessary preprocessing for validation.
        """
        cell = np.array(self.cell_sentences[idx], dtype=int)

        # Truncate or sample if the cell sentence exceeds max_length-2
        if len(cell) > self.max_length - 2:
            if self.is_random:
                sampled_indices = np.sort(np.random.choice(range(len(cell)), self.max_length - 2, replace=False))
                cell = cell[sampled_indices]
            else:
                cell = cell[:self.max_length - 2]

        # Add special tokens based on condition
        if self.condition[idx] == self.new_condition:
            cell = [self.num_cCRE + 4] + cell.tolist() + [2]
        else:
            cell = [1] + cell.tolist() + [2]

        # Convert to torch tensors
        cell_tensor = torch.tensor(cell, dtype=torch.long)

        return cell_tensor
    
class CellDatasetForDI(Dataset):
    """
    A PyTorch Dataset class for Data Imputation (DI) fine-tuning. This class processes input cell sentences
    and prepares them along with associated signals and data required for the Cell-cCRE Alignment (CCA) task.

    Args:
        adata (AnnData): Annotated data matrix containing the signal information for each cell.
        cell_sentences (list of list of int): List of cell sentences, where each cell sentence is a list of cCRE indices.
        max_length (int): Maximum length of the cell sentences, including special tokens [CLS] and [SEP] (default is 8192).
        alpha_for_CCA (int): Multiplier to determine the number of inaccessible cCREs sampled for the CCA task (default is 5).
        num_cCRE (int): Total number of candidate cis-regulatory elements (cCREs) (default is 1,355,445).
        is_random (bool): Whether to randomly sample indices if a cell sentence exceeds `max_length - 2` (default is False).

    Raises:
        ValueError: If `cell_sentences` is not properly formatted.
    """
    def __init__(
        self,
        adata,
        cell_sentences,
        max_length=8192,
        alpha_for_CCA=1,
        num_cCRE=1355445,
        is_random=False
    ):
        self.adata = adata.copy()
        self.max_length = max_length
        self.alpha_for_CCA = alpha_for_CCA
        self.is_random = is_random
        self.num_cCRE = num_cCRE

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
        Retrieves and processes a single cell sentence along with additional data required for UFE fine-tuning.

        Args:
            idx (int): Index of the cell sentence to retrieve.

        Returns:
            tuple:
                - cell_tensor (torch.Tensor): Processed cell sentence as a tensor of cCRE indices with special tokens.
                - signals_tensor (torch.Tensor): Signal values corresponding to the cCREs in the cell.
                - ex_cell_ccre_ids_for_CCA_tensor (torch.Tensor): Tensor of cCRE indices used in the CCA task.
                - ex_cell_ccre_accessibility_for_CCA_tensor (torch.Tensor): Tensor indicating accessibility labels for cCREs in the CCA task.
        """
        # Retrieve the cell sentence (list of cCRE indices)
        cell = np.array(self.cell_sentences[idx], dtype=int)

        # Retrieve the signal data for the cell from `adata`
        signals = self.adata[idx].X.toarray().reshape(-1)

        # ---------------------------------------------
        # Prepare data for the Cell-cCRE Alignment (CCA) task
        # ---------------------------------------------

        # Create a boolean array indicating all cCREs as inaccessible initially
        inaccessible_indices = np.ones(self.num_cCRE, dtype=bool)

        # Adjust cCRE indices if they start from an offset (assuming 4 as per the original code)
        adjusted_cell_indices = cell - 4  # Adjusting indices if necessary

        # Mark the accessible cCREs (those present in the cell) as False (i.e., not inaccessible)
        inaccessible_indices[adjusted_cell_indices] = False

        # Generate the list of inaccessible cCRE indices
        cell_inaccessible_indices = np.where(inaccessible_indices)[0] + 4  # Re-adjust indices if needed

        # Determine the number of inaccessible cCREs to sample based on `alpha_for_CCA`
        num_inaccessible_to_sample = int(len(cell) * self.alpha_for_CCA)

        # Randomly sample inaccessible cCREs without replacement
        if num_inaccessible_to_sample <= len(cell_inaccessible_indices):
            sampled_inaccessible_ccre = np.random.choice(
                cell_inaccessible_indices,
                size=num_inaccessible_to_sample,
                replace=False
            )
        else: 
            sampled_inaccessible_ccre = np.random.choice(
                cell_inaccessible_indices,
                size=num_inaccessible_to_sample,
                replace=True
            )

        # Combine accessible and sampled inaccessible cCRE indices for the CCA task
        ex_cell_ccre_ids_for_CCA = np.concatenate([cell, sampled_inaccessible_ccre])

        # Create accessibility labels: 1 for accessible cCREs, 0 for inaccessible cCREs
        ex_cell_ccre_accessibility_for_CCA = np.concatenate([
            np.ones_like(cell),
            np.zeros_like(sampled_inaccessible_ccre)
        ])

        # ---------------------------------------------
        # Truncate or sample the cell sentence if it exceeds `max_length - 2`
        # ---------------------------------------------

        # Check if the cell sentence needs to be truncated or sampled
        if len(cell) > self.max_length - 2:
            if self.is_random:
                # Randomly sample indices to reduce the length to `max_length - 2`
                sampled_indices = np.sort(
                    np.random.choice(
                        range(len(cell)),
                        size=self.max_length - 2,
                        replace=False
                    )
                )
                cell = cell[sampled_indices]
            else:
                # Truncate the cell sentence to the first `max_length - 2` elements
                cell = cell[:self.max_length - 2]

        # Add special tokens: [CLS] token ID is 1 at the beginning, [SEP] token ID is 2 at the end
        cell_with_special_tokens = [1] + cell.tolist() + [2]

        # ---------------------------------------------
        # Convert data to PyTorch tensors
        # ---------------------------------------------

        cell_tensor = torch.tensor(cell_with_special_tokens, dtype=torch.long)
        signals_tensor = torch.tensor(signals, dtype=torch.float32)
        ex_cell_ccre_ids_for_CCA_tensor = torch.tensor(ex_cell_ccre_ids_for_CCA, dtype=torch.long)
        ex_cell_ccre_accessibility_for_CCA_tensor = torch.tensor(ex_cell_ccre_accessibility_for_CCA, dtype=torch.float32)

        return (
            cell_tensor,
            signals_tensor,
            ex_cell_ccre_ids_for_CCA_tensor,
            ex_cell_ccre_accessibility_for_CCA_tensor
        )


def collate_fn_for_DI(data):
    """
    Collate function for padding sequences and preparing batches for DI fine-tuning.

    This function pads the cell sequences to the same length within a batch and
    compiles the necessary data for the Cell-cCRE Alignment (CCA) task.

    Args:
        data (list of tuples): Each tuple contains:
            - cell_tensor (torch.Tensor): Processed cell sequence tensor.
            - signals_tensor (torch.Tensor): Signal values tensor.
            - ex_cell_ccre_ids_for_CCA_tensor (torch.Tensor): cCRE indices tensor for the CCA task.
            - ex_cell_ccre_accessibility_for_CCA_tensor (torch.Tensor): Accessibility labels tensor for the CCA task.

    Returns:
        tuple:
            - padded_cells_batch (torch.Tensor): Padded batch of cell sequences.
            - signals_batch (torch.Tensor): Batch of signal arrays.
            - ex_cell_ccre_ids_batch (list of torch.Tensor): List of cCRE indices tensors for the CCA task for each sample.
            - ex_cell_ccre_accessibility_batch (torch.Tensor): Concatenated accessibility labels tensor for the CCA task.
    """
    # Unzip the data into individual components
    cells_list, signals_list, ex_cell_ccre_ids_list, ex_cell_ccre_accessibility_list = zip(*data)

    # Determine the maximum sequence length in the batch
    max_seq_length = max([cell.size(0) for cell in cells_list])

    # Pad the cell sequences to the maximum length using the PAD token ID (0)
    padded_cells_batch = torch.stack([
        torch.nn.functional.pad(
            cell,
            pad=(0, max_seq_length - cell.size(0)),
            mode='constant',
            value=0  # PAD token ID is 0
        ) for cell in cells_list
    ])

    # Stack the signals into a batch tensor
    signals_batch = torch.stack(signals_list)

    # Collect cCRE IDs for the CCA task into a list (they can have variable lengths)
    ex_cell_ccre_ids_batch = list(ex_cell_ccre_ids_list)

    # Concatenate accessibility labels for the CCA task into a single tensor
    ex_cell_ccre_accessibility_batch = torch.cat(ex_cell_ccre_accessibility_list)

    return (
        padded_cells_batch,
        signals_batch,
        ex_cell_ccre_ids_batch,
        ex_cell_ccre_accessibility_batch
    )


class CellDatasetForRDI(Dataset):
    """
    A PyTorch Dataset class for handling cell sentences and related data in the task of reference data integration (RDI).
    This class processes input cell sentences, with predicted cell sentences and batch-specific information,
    and prepares them for downstream tasks such as matching cells across batches.

    Args:
        adata (AnnData): Annotated data matrix containing the signal information for each cell.
        cell_sentences (list of list of int): List of cell sentences, where each cell sentence is a list of cCRE indices.
        predicted_cell_sentences (list of list of int): List of predicted cell sentences, where each predicted sentence corresponds to a list of cCRE indices.
        max_length (int): Maximum length of the cell sentences, including special tokens [CLS] and [SEP] (default is 8192).
        alpha_for_CCA (int): Multiplier to determine the number of inaccessible cCREs sampled for the CCA task (default is 1).
        num_cCRE (int): Total number of candidate cis-regulatory elements (cCREs) (default is 1,355,445).
        is_random (bool): Whether to randomly sample indices if a cell sentence exceeds max_length-2 (default is False).
        batch_ids (list of int): List of batch identifiers for each cell (must be integers).
    
    Raises:
        ValueError: If cell_sentences, predicted_cell_sentences, or other arguments are not properly formatted.
    """
    
    def __init__(self, 
                 adata, 
                 cell_sentences, 
                 predicted_cell_sentences, 
                 max_length=8192, 
                 alpha_for_CCA=1, 
                 num_cCRE=1355445, 
                 is_random=False, 
                 batch_ids=None):
        self.max_length = max_length
        self.is_random = is_random
        self.alpha_for_CCA = alpha_for_CCA
        self.num_cCRE = num_cCRE
        self.adata = adata
        self.batch_ids = batch_ids
        
        # Ensure batch_ids is provided
        if batch_ids is None:
            raise ValueError("`batch_ids` must be provided for RDI task.")
        
        # Make sure the batch_ids match the number of cells in the dataset
        if len(batch_ids) != len(cell_sentences):
            raise ValueError("The length of `batch_ids` must match the number of cells.")
        
        # Process the cell sentences and predicted sentences
        self.cell_sentences = cell_sentences
        self.predicted_cell_sentences = predicted_cell_sentences

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

        # Ensure predicted_cell_sentences is a list and parse JSON strings
        try:
            self.predicted_cell_sentences = [
                json.loads(instance) if isinstance(instance, str) else instance
                for instance in predicted_cell_sentences
            ]
        except (json.JSONDecodeError, TypeError):
            raise ValueError("Error parsing cell_sentences. Ensure all string elements are JSON serializable lists.")

        # Validate format of predicted_cell_sentences
        if not all(isinstance(sentence, list) and all(isinstance(i, int) for i in sentence) for sentence in self.predicted_cell_sentences):
            raise ValueError("Invalid format for cell_sentences. Each element must be a list of integers.")

        
        # --- Step 1: Modify the input adata.X ---
        # Set all positive values in adata.X to 1 (binary encoding for presence)
        self.adata.X[self.adata.X > 0] = 1

    def __len__(self):
        """
        Returns the number of cell sentences in the dataset.

        Returns:
            int: Number of cell sentences.
        """
        return len(self.cell_sentences)

    def __getitem__(self, idx):
        """
        Retrieves and processes a single cell sentence along with additional data required for RDI.

        Args:
            idx (int): Index of the cell sentence to retrieve.

        Returns:
            tuple:
                - cell_tensor (torch.Tensor): Processed cell sentence as a tensor of cCRE indices with special tokens.
                - signals_tensor (torch.Tensor): Signal values corresponding to the cCREs in the cell.
                - batch_labels_tensor (torch.Tensor): Tensor for batch labels corresponding to each cell.
                - ex_cell_ccre_ids_for_CCA_tensor (torch.Tensor): Tensor for the cCRE indices used in the Cell-cCRE Alignment (CCA) task.
                - ex_cell_ccre_accessibility_for_CCA_tensor (torch.Tensor): Tensor for the accessibility labels for cCREs.
        """
        # Retrieve the cell sentence (list of cCRE indices)
        cell = np.array(self.cell_sentences[idx], dtype=int)
        predicted_cell = np.array(self.predicted_cell_sentences[idx], dtype=int)

        # Retrieve the signal data for the cell from `adata`
        signals = self.adata[idx].X.toarray().reshape(-1)

        # ---------------------------------------------
        # Prepare data for the Cell-cCRE Alignment (CCA) task
        # ---------------------------------------------

        # Create a boolean array indicating all cCREs as inaccessible initially
        predicted_inaccessible_indices = np.ones(self.num_cCRE, dtype=bool)

        # Adjust cCRE indices if they start from an offset (assuming 4 as per the original code)
        adjusted_cell_indices = cell - 4  # Adjusting indices if necessary
        predicted_adjusted_cell_indices = predicted_cell - 4

        # Mark the accessible cCREs (those present in the cell) as False (i.e., not inaccessible)
        predicted_inaccessible_indices[predicted_adjusted_cell_indices] = False

        # Generate the list of inaccessible cCRE indices
        predicted_cell_inaccessible_indices = np.where(predicted_inaccessible_indices)[0] + 4  # Re-adjust indices if needed

        # Determine the number of inaccessible cCREs to sample based on `alpha_for_CCA`
        num_inaccessible_to_sample = int(len(predicted_cell) * self.alpha_for_CCA)

        # Randomly sample inaccessible cCREs without replacement
        if num_inaccessible_to_sample <= len(predicted_cell_inaccessible_indices):
            predicted_sampled_inaccessible_ccre = np.random.choice(
                predicted_cell_inaccessible_indices,
                size=num_inaccessible_to_sample,
                replace=False
            )
        else: 
            predicted_sampled_inaccessible_ccre = np.random.choice(
                predicted_cell_inaccessible_indices,
                size=num_inaccessible_to_sample,
                replace=True
            )

        # Combine accessible and sampled inaccessible cCRE indices for the CCA task
        ex_cell_ccre_ids_for_CCA = np.concatenate([predicted_cell, predicted_sampled_inaccessible_ccre])

        # Create accessibility labels: 1 for accessible cCREs, 0 for inaccessible cCREs
        ex_cell_ccre_accessibility_for_CCA = np.concatenate([ 
            np.ones_like(predicted_cell),
            np.zeros_like(predicted_sampled_inaccessible_ccre)
        ])

        # ---------------------------------------------
        # Truncate or sample the cell sentence if it exceeds `max_length - 2`
        # ---------------------------------------------

        # Check if the cell sentence needs to be truncated or sampled
        if len(cell) > self.max_length - 2:
            if self.is_random:
                # Randomly sample indices to reduce the length to `max_length - 2`
                sampled_indices = np.sort(
                    np.random.choice(
                        range(len(cell)),
                        size=self.max_length - 2,
                        replace=False
                    )
                )
                cell = cell[sampled_indices]
            else:
                # Truncate the cell sentence to the first `max_length - 2` elements
                cell = cell[:self.max_length - 2]

        # Add special tokens: [CLS] token ID is 1 at the beginning, [SEP] token ID is 2 at the end
        cell = [1] + cell.tolist() + [2]

        # Convert to torch tensors
        cell_tensor = torch.tensor(cell, dtype=torch.long)
        signals_tensor = torch.tensor(signals, dtype=torch.float32)
        ex_cell_ccre_ids_for_CCA_tensor = torch.tensor(ex_cell_ccre_ids_for_CCA, dtype=torch.long)
        ex_cell_ccre_accessibility_for_CCA_tensor = torch.tensor(ex_cell_ccre_accessibility_for_CCA, dtype=torch.float32)
        
        # Return batch labels (batch_ids corresponding to this index)
        batch_labels_tensor = torch.tensor(self.batch_ids[idx], dtype=torch.long)

        return (
            cell_tensor,
            signals_tensor,
            batch_labels_tensor,
            ex_cell_ccre_ids_for_CCA_tensor,
            ex_cell_ccre_accessibility_for_CCA_tensor
        )

def collate_fn_for_RDI(data):
    """
    Collate function for padding sequences and preparing batches for the RDI task.

    This function pads the cell sequences to the same length within a batch and
    compiles the necessary data for the Cell-cCRE Alignment (CCA) task.

    Args:
        data (list of tuples): Each tuple contains:
            - torch.Tensor: Processed cell sentence tensor.
            - torch.Tensor: Signal values tensor.
            - torch.Tensor: Classification label tensor.
            - torch.Tensor: Tensor for the cCRE indices used in the CCA task.
            - torch.Tensor: Tensor for the accessibility labels of the cCREs.

    Returns:
        tuple: A tuple containing:
            - padded_cells_batch (torch.Tensor): Padded batch of cell sequences.
            - signals_batch (torch.Tensor): Batch of signal arrays.
            - classification_labels_batch (torch.Tensor): Batch of classification labels.
            - ex_cell_ccre_ids_for_CCA_batch (list of torch.Tensor): List of tensors for CCA task.
            - ex_cell_ccre_accessibility_for_CCA_batch (torch.Tensor): Concatenated tensor for CCA task accessibility labels.
    """
    # Unzip the data into individual components
    cells_list, signals_list, classification_labels_list, ex_cell_ccre_ids_for_CCA_list, ex_cell_ccre_accessibility_for_CCA_list = zip(*data)

    # Determine the maximum sequence length in the batch
    max_seq_length = max([cell.size(0) for cell in cells_list])

    # Pad the cell sequences to the maximum length using the PAD token ID (0)
    padded_cells_batch = torch.stack([
        torch.nn.functional.pad(
            cell,
            pad=(0, max_seq_length - cell.size(0)),
            mode='constant',
            value=0  # PAD token ID is 0
        ) for cell in cells_list
    ])

    # Stack the signals and classification labels into batch tensors
    signals_batch = torch.stack(signals_list)
    classification_labels_batch = torch.stack(classification_labels_list)

    # Collect the ex_cell_ccre_ids_for_CCA tensors (for CCA task) into a list
    ex_cell_ccre_ids_for_CCA_batch = list(ex_cell_ccre_ids_for_CCA_list)

    # Concatenate ex_cell_ccre_accessibility_for_CCA into a single tensor
    ex_cell_ccre_accessibility_for_CCA_batch = torch.cat(ex_cell_ccre_accessibility_for_CCA_list)

    return padded_cells_batch, signals_batch, classification_labels_batch, ex_cell_ccre_ids_for_CCA_batch, ex_cell_ccre_accessibility_for_CCA_batch
class TestCellDatasetForRDI(Dataset):
    """
    A PyTorch Dataset class for handling cell sentences and related data in the task of reference data integration (RDI).
    This class processes input cell sentences, with predicted cell sentences and batch-specific information,
    and prepares them for downstream tasks such as matching cells across batches.

    Args:
        adata (AnnData): Annotated data matrix containing the signal information for each cell.
        cell_sentences (list of list of int): List of cell sentences, where each cell sentence is a list of cCRE indices.
        predicted_cell_sentences (list of list of int): List of predicted cell sentences, where each predicted sentence corresponds to a list of cCRE indices.
        max_length (int): Maximum length of the cell sentences, including special tokens [CLS] and [SEP] (default is 8192).
        alpha_for_CCA (int): Multiplier to determine the number of inaccessible cCREs sampled for the CCA task (default is 1).
        num_cCRE (int): Total number of candidate cis-regulatory elements (cCREs) (default is 1,355,445).
        is_random (bool): Whether to randomly sample indices if a cell sentence exceeds max_length-2 (default is False).
        batch_ids (list of int): List of batch identifiers for each cell (must be integers).
    
    Raises:
        ValueError: If cell_sentences, predicted_cell_sentences, or other arguments are not properly formatted.
    """
    
    def __init__(self, 
                 cell_sentences, 
                 max_length=8192, 
                 is_random=False, 
                 batch_ids=None):
        self.max_length = max_length
        self.is_random = is_random
        self.batch_ids = batch_ids
        
        # Ensure batch_ids is provided
        if batch_ids is None:
            raise ValueError("`batch_ids` must be provided for inference.")
        
        # Make sure the batch_ids match the number of cells in the dataset
        if len(batch_ids) != len(cell_sentences):
            raise ValueError("The length of `batch_ids` must match the number of cells.")
        
        # Process the cell sentences and predicted sentences
        self.cell_sentences = cell_sentences

    
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


    def __len__(self):
        """
        Returns the number of cell sentences in the dataset.

        Returns:
            int: Number of cell sentences.
        """
        return len(self.cell_sentences)

    def __getitem__(self, idx):
        """
        Retrieves and processes a single cell sentence along with additional data required for RDI.

        Args:
            idx (int): Index of the cell sentence to retrieve.

        Returns:
            tuple:
                - cell_tensor (torch.Tensor): Processed cell sentence as a tensor of cCRE indices with special tokens.
                - signals_tensor (torch.Tensor): Signal values corresponding to the cCREs in the cell.
                - batch_labels_tensor (torch.Tensor): Tensor for batch labels corresponding to each cell.
                - ex_cell_ccre_ids_for_CCA_tensor (torch.Tensor): Tensor for the cCRE indices used in the Cell-cCRE Alignment (CCA) task.
                - ex_cell_ccre_accessibility_for_CCA_tensor (torch.Tensor): Tensor for the accessibility labels for cCREs.
        """
        # Retrieve the cell sentence (list of cCRE indices)
        cell = np.array(self.cell_sentences[idx], dtype=int)
       
        # ---------------------------------------------
        # Truncate or sample the cell sentence if it exceeds `max_length - 2`
        # ---------------------------------------------

        # Check if the cell sentence needs to be truncated or sampled
        if len(cell) > self.max_length - 2:
            if self.is_random:
                # Randomly sample indices to reduce the length to `max_length - 2`
                sampled_indices = np.sort(
                    np.random.choice(
                        range(len(cell)),
                        size=self.max_length - 2,
                        replace=False
                    )
                )
                cell = cell[sampled_indices]
            else:
                # Truncate the cell sentence to the first `max_length - 2` elements
                cell = cell[:self.max_length - 2]

        # Add special tokens: [CLS] token ID is 1 at the beginning, [SEP] token ID is 2 at the end
        cell = [1] + cell.tolist() + [2]

        # Convert to torch tensors
        cell_tensor = torch.tensor(cell, dtype=torch.long)

        # Return batch labels (batch_ids corresponding to this index)
        batch_labels_tensor = torch.tensor(self.batch_ids[idx], dtype=torch.long)

        return (
            cell_tensor,
            batch_labels_tensor
        )
def collate_fn_for_RDI_test(data):
    """
    Collate function for padding sequences and preparing batches for the RDI inference.

    This function pads the cell sequences to the same length within a batch and
    compiles the necessary data for the Cell-cCRE Alignment (CCA) task.

    Args:
        data (list of tuples): Each tuple contains:
            - torch.Tensor: Processed cell sentence tensor.
            - torch.Tensor: Batch label tensor.

    Returns:
        tuple: A tuple containing:
            - padded_cells_batch (torch.Tensor): Padded batch of cell sequences.
            - classification_labels_batch (torch.Tensor): Batch labels.
    """
    # Unzip the data into individual components
    cells_list, batch_labels_list = zip(*data)

    # Determine the maximum sequence length in the batch
    max_seq_length = max([cell.size(0) for cell in cells_list])

    # Pad the cell sequences to the maximum length using the PAD token ID (0)
    padded_cells_batch = torch.stack([
        torch.nn.functional.pad(
            cell,
            pad=(0, max_seq_length - cell.size(0)),
            mode='constant',
            value=0  # PAD token ID is 0
        ) for cell in cells_list
    ])

    # Stack the batch labels into batch tensors
    batch_labels_batch = torch.stack(batch_labels_list)

    return padded_cells_batch, batch_labels_batch
