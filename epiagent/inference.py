import torch
import numpy as np
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from collections import Counter

def infer_reconstructed_signals(model, device, dataloader, need_cell_embeddings=True, predicted_cCRE_indices=None):
    """
    Performs inference using the EpiAgent model to predict reconstructed signals and optionally extract cell embeddings.

    Args:
        model (nn.Module): The EpiAgent model.
        device (torch.device): The device to run the inference on (e.g., 'cuda' or 'cpu').
        dataloader (DataLoader): The DataLoader providing batches of input data.
        need_cell_embeddings (bool): Whether to return cell embeddings. Default is True.
        predicted_cCRE_indices (numpy.ndarray, optional): A 1D array of cCRE indices to predict. If None, the model predicts accessibility for all cCREs. 
            Default is None.

    Returns:
        dict: A dictionary containing:
            - 'predicted_signals' (numpy.ndarray): The predicted signals for each cell.
            - 'cell_embeddings' (numpy.ndarray, optional): The cell embeddings if `need_cell_embeddings` is True.
    """
    # Move model to the specified device and set to evaluation mode
    model.to(device)
    model.eval()

    # Initialize storage for predicted signals and optional embeddings
    predicted_signals = []
    cell_embeddings = [] if need_cell_embeddings else None

    # Perform inference
    for batch in dataloader:
        # Clear GPU cache
        torch.cuda.empty_cache()

        # Move input batch to the specified device
        input_ids = batch.to(device)

        # Enable mixed precision for faster inference if supported
        with autocast():  # Enable mixed precision for faster inference
            with torch.no_grad():  # Disable gradient computation for inference
                # Get model outputs: transformer outputs and predicted signals
                output = model(input_ids, return_transformer_output=True, return_SD_output=True)

                # Extract the transformer outputs (cell embeddings)
                transformer_outputs = output['transformer_outputs'][:, 0, :]
                
                # Optionally collect the cell embeddings
                if need_cell_embeddings:
                    embeddings = transformer_outputs.cpu().detach().numpy()
                    cell_embeddings.extend(embeddings)

                # Get the predicted signals from the model output
                predicted_signal_batch = output['predicted_signals'].cpu().detach().numpy()
                
                # If predicted_cCRE_indices is provided, filter the predicted signals for those cCREs
                if predicted_cCRE_indices is not None:
                    predicted_signal_batch = predicted_signal_batch[:, predicted_cCRE_indices]

                predicted_signals.extend(predicted_signal_batch)

    # Convert the list of predictions and embeddings to numpy arrays
    predicted_signals = np.array(predicted_signals)

    # Prepare the output dictionary
    outputs = {
        'predicted_signals': predicted_signals
    }

    # Include the cell embeddings if requested
    if need_cell_embeddings:
        cell_embeddings = np.array(cell_embeddings)
        outputs['cell_embeddings'] = cell_embeddings

    return outputs

def infer_cell_embeddings(model, device, dataloader):
    """
    Performs inference using the EpiAgent model to extract cell embeddings.

    Args:
        model (nn.Module): The EpiAgent model.
        device (torch.device): The device to run the inference on (e.g., 'cuda' or 'cpu').
        dataloader (DataLoader): The DataLoader providing batches of input data.

    Returns:
        numpy.ndarray: A numpy array of cell embeddings with shape [num_cells, embedding_dim].
    """
    # Move model to the specified device and set to evaluation mode
    model.to(device)
    model.eval()

    # Initialize storage for embeddings
    cell_embeddings = []

    # Perform inference
    for batch in dataloader:
        # Clear GPU cache
        torch.cuda.empty_cache()

        # Move input batch to the specified device
        input_ids = batch.to(device)
        with autocast():  # Enable mixed precision for faster inference
            with torch.no_grad():  # Disable gradient computation for inference
                # Get transformer outputs (cell embeddings)
                output = model(input_ids, return_transformer_output=True)
                embeddings = output['transformer_outputs'][:, 0, :]
    
                # Move embeddings to CPU and detach for storage
                embeddings = embeddings.cpu().detach().numpy()
                cell_embeddings.extend(embeddings)

    # Convert the list of embeddings to a numpy array
    return np.array(cell_embeddings)

def infer_cell_types(model, device, dataloader, need_cell_embeddings=True):
    """
    Performs inference using the EpiAgent_supervised model to predict cell types and extract embeddings.

    Args:
        model (nn.Module): The EpiAgent_supervised model.
        device (torch.device): The device to run the inference on (e.g., 'cuda' or 'cpu').
        dataloader (DataLoader): The DataLoader providing batches of input data.
        need_cell_embeddings (bool): If True, returns cell embeddings in addition to predictions (default is True).

    Returns:
        dict: A dictionary containing:
            - 'predicted_labels': A list of predicted probabilities for each cell.
            - 'predicted_classes': A list of predicted cell types (class indices).
            - 'cell_embeddings' (optional): A numpy array of cell embeddings, if need_cell_embeddings is True.
    """
    # Move model to the specified device and set to evaluation mode
    model.to(device)
    model.eval()

    # Initialize storage for results
    cell_embeddings = []
    predicted_probabilities = []
    predicted_classes = []

    # Perform inference
    for batch in dataloader:
        # Clear GPU cache
        torch.cuda.empty_cache()

        # Move input batch to the specified device
        input_ids = batch.to(device)

        with autocast():  # Enable mixed precision for faster inference
            # Get model predictions and optionally cell embeddings
            if need_cell_embeddings:
                predicted_label, embeddings = model(input_ids, need_cell_embeddings=True)
                embeddings = embeddings.cpu().detach().numpy()
                cell_embeddings.extend(embeddings)
            else:
                predicted_label = model(input_ids, need_cell_embeddings=False)

            # Convert predictions to probabilities and class indices
            probabilities = torch.softmax(predicted_label, dim=1)
            _, predicted_indices = torch.max(probabilities, dim=1)

            # Store results
            predicted_probabilities.extend(probabilities.cpu().detach().numpy())
            predicted_classes.extend(predicted_indices.cpu().detach().numpy().tolist())

    # Prepare the result dictionary
    results = {
        'predicted_probabilities': predicted_probabilities,
        'predicted_labels': predicted_classes
    }
    if need_cell_embeddings:
        results['cell_embeddings'] = np.array(cell_embeddings)

    return results

def filter_rare_cell_types(predicted_cell_type, predicted_label_list, threshold=0.005):
    """
    Filter predicted cell types based on occurrence frequency.

    This function is specifically designed for EpiAgent-B and EpiAgent-NT models to filter out
    predicted cell types that occur less frequently than the specified threshold.

    Args:
        predicted_cell_type (list): List of predicted cell types.
        predicted_label_list (list of ndarray): List of predicted probabilities for each cell.
        threshold (float): The minimum frequency threshold for filtering (default is 0.005).

    Returns:
        list: Filtered predicted cell types.
    """
    # Count occurrences of each cell type
    counter = Counter(predicted_cell_type)
    total_count = len(predicted_cell_type)

    # Identify rare cell types
    rare_elements = [element for element, count in counter.items() if (count / total_count) < threshold]

    # Filter predictions by setting rare elements to -inf
    predicted_cell_type_filtered = []
    for predicted_label in predicted_label_list:
        predicted_label[np.isin(np.arange(len(predicted_label)), rare_elements)] = -float('inf')
        predicted_classes = np.argmax(predicted_label, axis=0)
        predicted_cell_type_filtered.append(predicted_classes)

    return predicted_cell_type_filtered


def infer_cell_embeddings_for_RDI(model, device, dataloader):
    """
    Performs inference using the fine-tuned EpiAgent_BC model to extract cell embeddings for reference data integration (RDI).

    Args:
        model (nn.Module): The fine-tuned EpiAgent_BC model.
        device (torch.device): The device to run the inference on (e.g., 'cuda' or 'cpu').
        dataloader (DataLoader): The DataLoader providing batches of input data.

    Returns:
        numpy.ndarray: A numpy array of cell embeddings with shape [num_cells, embedding_dim].
    """
    # Move model to the specified device and set to evaluation mode
    model.to(device)
    model.eval()

    # Initialize storage for embeddings
    cell_embeddings = []

    # Perform inference
    for batch in dataloader:
        # Clear GPU cache
        torch.cuda.empty_cache()

        # Move input batch to the specified device
        input_ids = batch[0].to(device)
        batch_ids = batch[1].to(device)

        with autocast():  # Enable mixed precision for faster inference
            with torch.no_grad():  # Disable gradient computation for inference
                # Get transformer outputs (cell embeddings)
                output = model(input_ids, batch_ids, return_transformer_output=True)
                embeddings = output['transformer_outputs'][:, 0, :]  # Get embeddings from the [CLS] token

                # Move embeddings to CPU and detach for storage
                embeddings = embeddings.cpu().detach().numpy()
                cell_embeddings.extend(embeddings)

    # Convert the list of embeddings to a numpy array
    return np.array(cell_embeddings)