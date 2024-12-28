import numpy as np
import pandas as pd
from anndata import AnnData


def tokenization(adata: AnnData):
    """
    Tokenization function for an AnnData object.

    This function performs several checks and processes the AnnData object to generate tokenized cell sentences
    based on cCRE indices and stores them in `adata.obs['cell_sentences']`.

    Args:
        adata (AnnData): Input AnnData object.

    Raises:
        ValueError: If feature dimensions are incorrect.
        ValueError: If features are not continuous (TF-IDF not applied).
        ValueError: If `adata.X` is not in the expected sparse matrix format.
    """
    # Step 1: Check the number of features in adata.X
    if adata.shape[1] != 1355445:
        raise ValueError("Feature dimensions are not 1355445. Please ensure you are using EpiAgent-required cCREs.")

    # Step 2: Check if the data is continuous (TF-IDF applied)
    if not np.issubdtype(adata.X.dtype, np.floating):
        raise ValueError("Feature values are not continuous. Please apply the global_TFIDF function before tokenization.")

    # Step 3: Check if 'cell_sentences' already exists in adata.obs
    if 'cell_sentences' in adata.obs.columns:
        print("Warning: 'cell_sentences' column already exists in adata.obs. The existing column will be overwritten.")

    # Step 4: Check if `adata.X` is in the expected sparse matrix format
    if hasattr(adata.X, 'toarray'):
        if not isinstance(adata.X, np.ndarray):
            cell_sentences = []

            # Iterate through each cell in the dataset
            for cell in adata:
                # Convert sparse matrix to dense array and flatten
                cell_array = cell.X.toarray().reshape(-1)

                # Get indices of non-zero values and sort them in descending order by value
                non_zero_indices = np.nonzero(cell_array)[0]
                sorted_indices = non_zero_indices[np.argsort(-cell_array[non_zero_indices])] + 4

                # Convert indices to string format to avoid errors when saving AnnData
                sorted_indices_str = "["+",".join(map(str, sorted_indices))+"]"
                cell_sentences.append(sorted_indices_str)  # Append to cell_sentences list

            # Add cell_sentences to adata.obs
            adata.obs['cell_sentences'] = pd.Series(cell_sentences, index=adata.obs.index)
            print("Tokenization complete: 'cell_sentences' column added to adata.obs.")
        else:
            raise ValueError("`adata.X` is not in the expected sparse matrix format. Please check the input data.")
    else:
        # If the matrix is not sparse, process directly without `toarray`
        cell_sentences = []
        for cell in adata:
            cell_array = cell.X.reshape(-1)
            non_zero_indices = np.nonzero(cell_array)[0]
            sorted_indices = non_zero_indices[np.argsort(-cell_array[non_zero_indices])] + 4
            sorted_indices_str = "["+",".join(map(str, sorted_indices))+"]"
            cell_sentences.append(sorted_indices_str)

        # Add cell_sentences to adata.obs
        adata.obs['cell_sentences'] = pd.Series(cell_sentences, index=adata.obs.index)
        print("Tokenization complete: 'cell_sentences' column added to adata.obs.")