import numpy as np
from anndata import AnnData
import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix

def construct_cell_by_ccre_matrix(intersect_file, ccre_bed_path):
    """
    Constructs a cell-by-cCRE matrix from intersect results and cCRE definitions.

    Args:
        intersect_file (str): Path to the intersect result file.
        ccre_bed_path (str): Path to the cCRE definition file.

    Returns:
        AnnData: An AnnData object containing the cell-by-cCRE matrix.
    """
    # Read cCRE definitions
    ccre_df = pd.read_csv(ccre_bed_path, sep='\t', header=None, names=['chrom', 'start', 'end'])
    ccre_df['ccre_detail'] = ccre_df['chrom'].astype(str) + ':' + ccre_df['start'].astype(str) + '-' + ccre_df['end'].astype(str)
    ccre_details = pd.Series(ccre_df['ccre_detail'].values, index=ccre_df['ccre_detail']).to_dict()

    # Read intersect file
    intersect_df = pd.read_csv(intersect_file, sep='\t', header=None)

    # Determine format and extract overlap information
    if str(intersect_df.iloc[0, 4]).startswith('chr'):
        print("Detected no count information in intersect file. Using binary overlap (count = 1).")
        intersect_ccre_details = intersect_df.iloc[:, 4].astype(str) + ':' + intersect_df.iloc[:, 5].astype(str) + '-' + intersect_df.iloc[:, 6].astype(str)
        overlaps = np.ones(intersect_df.shape[0], dtype=np.float32)
    else:
        intersect_ccre_details = intersect_df.iloc[:, 5].astype(str) + ':' + intersect_df.iloc[:, 6].astype(str) + '-' + intersect_df.iloc[:, 7].astype(str)
        overlaps = intersect_df[4].values.astype(np.float32)

    # Map cells and cCREs to categorical indices
    cells = intersect_df[3]
    used_cells = np.unique(cells).astype(str)
    cell_ids = pd.Categorical(cells, categories=used_cells).codes
    ccre_ids = pd.Categorical(intersect_ccre_details, categories=ccre_details).codes

    # Create sparse matrix
    cell_ccre_matrix = coo_matrix((overlaps, (cell_ids, ccre_ids)), shape=(len(used_cells), len(ccre_details)), dtype=np.float32).tocsr()

    # Create AnnData
    adata = AnnData(X=cell_ccre_matrix)
    adata.obs_names = used_cells
    adata.var_names = pd.Series(list(ccre_details.keys()))

    return adata

def global_TFIDF(adata, cCRE_document_frequency):
    """
    Apply a global TF-IDF transformation to the input AnnData object.
    
    Parameters:
    - adata: AnnData
        The input AnnData object containing the sparse matrix in `adata.X`.
    - cCRE_document_frequency: ndarray
        A 1D numpy array representing the document frequency for each cCRE (column).

    Returns:
    - AnnData
        A new AnnData object with TF-IDF-transformed data in `adata.X`.
    """
    # Create a copy of the input AnnData object to avoid modifying the original
    adata_copy = adata.copy()

    # Step 1: Compute the reciprocal of column sums (document frequency)
    cCRE_doc_frequency_inv = csr_matrix(np.reciprocal(cCRE_document_frequency))

    # Step 2: Compute row sums (total counts per cell)
    row_sums = adata_copy.X.sum(axis=1)
    row_sums_inv = csr_matrix(np.reciprocal(row_sums.A.flatten()))  # Convert to CSR matrix for sparse multiplication

    # Step 3: Normalize rows by row sums
    adata_copy.X = adata_copy.X.multiply(row_sums_inv.T)

    # Step 4: Normalize columns by document frequency
    adata_copy.X = adata_copy.X.multiply(cCRE_doc_frequency_inv)

    # Step 5: Apply log transformation to all non-zero elements
    adata_copy.X.data = np.log1p(10000 * adata_copy.X.data)

    return adata_copy