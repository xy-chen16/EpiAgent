import scanpy as sc
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from epiagent.tokenization import tokenization
from epiagent.preprocessing import global_TFIDF
from epiagent.dataset import CellDatasetForDI, collate_fn_for_DI
from epiagent.dataset import CellDataset, collate_fn
from epiagent.model import EpiAgent
from epiagent.train import fine_tune_epiagent_for_UFE
from epiagent.inference import infer_reconstructed_signals

def evaluation_pipeline(dataset_name, raw_h5ad_path, imputed_h5ad_path, open_threshold=0.03, celltype_key='cell_type', output_path='./', reference_csv_path=None, plot=False):

    # 1. Load data
    raw_adata = sc.read_h5ad(raw_h5ad_path)

    cCRE_frequency_path = '/user/work/likeyi/EpiAgent/20260520_re_imputation_on_B2018/data/cCRE_document_frequency.npy'
    cCRE_document_frequency = np.load(cCRE_frequency_path)

    # Apply TF-IDF transformation
    raw_adata = global_TFIDF(raw_adata, cCRE_document_frequency)

    adata = sc.read_h5ad(imputed_h5ad_path)
    print(f"Initial data shape: {adata.shape}")

    # 2. Feature selection
    cell_counts_per_cre = np.array((raw_adata.X > 0).sum(axis=0)).flatten()
    min_cells = raw_adata.shape[0] * open_threshold
    select_peak = cell_counts_per_cre >= min_cells
    raw_adata = raw_adata[:, select_peak].copy()

    if adata.n_vars == raw_adata.n_vars:
        print(f"Feature has already been selected: {adata.n_vars} features.")
    elif adata.n_vars == len(select_peak):
        adata = adata[:, select_peak].copy()
        print(f"Selected {adata.n_vars} features open in >= {open_threshold*100}% cells.")
    else:
        raise ValueError(f"Wrong features! {adata.n_vars} != {len(select_peak)}")

    # 3. Reference Building
    celltypes = adata.obs[celltype_key].unique()
    
    if reference_csv_path:
        mean_signals_dict = pd.read_csv(reference_csv_path, index_col=0)
    else:
        print("Calculating mean signals per cell type...")
        mean_signals_dict = {}
        for ct in celltypes:
            ct_mask = raw_adata.obs[celltype_key] == ct
            sub_matrix = raw_adata[ct_mask].X
            if sp.issparse(sub_matrix):
                sub_matrix = sub_matrix.toarray()
            mean_signals_dict[ct] = sub_matrix.mean(axis=0)
    
    # 4. Calculating correlation
    all_correlations = np.zeros(adata.shape[0])
    
    print("Calculating Pearson correlation cell-wise...")
    for ct in tqdm(celltypes):
        mask = (adata.obs[celltype_key] == ct).values
        idx = np.where(mask)[0]
        
        X_sub = adata.X[mask]
        if sp.issparse(X_sub):
            X_sub = X_sub.toarray()

        y = mean_signals_dict[ct]
        if isinstance(y, pd.Series): y = y.values

        corr_ct = []
        for cell_profile in X_sub:
            r, _ = pearsonr(cell_profile, y)
            corr_ct.append(r)

        all_correlations[idx] = corr_ct

    # 5. Save the result
    np.save(f"{output_path}/{dataset_name}_pearson_results.npy", all_correlations)
    print(f"Evaluation finished. Median Correlation: {np.median(all_correlations):.4f}")

    if plot:
        plt.figure(figsize=(4, 5))
        sns.violinplot(y=all_correlations, color="#5c9eb2", inner="box", cut=0)

        plt.title(f"{dataset_name} - Overall Pearson Correlation", fontsize=12)
        plt.ylabel("Pearson r", fontsize=10)
        plt.legend(loc="upper left")
        plt.tight_layout()

        plt.savefig(f"{output_path}/{dataset_name}_pearson_overall_violin.png", dpi=300)
        plt.close()
        print(f"Violin plot saved to: {dataset_name}_pearson_overall_violin.png")

def main():
    # 1. Data
    # Load the raw data
    input_path = '/user/work/likeyi/EpiAgent/20260520_re_imputation_on_B2018/data/Buenrostro2018-bone_marrow_tissue-cell_by_cCRE.h5ad'
    print(f"Loading data from path: {input_path}...")
    adata = sc.read_h5ad(input_path)
    print(f"Loaded adata: {adata}")

    # Load the cCRE document frequency data
    cCRE_frequency_path = '/user/work/likeyi/EpiAgent/20260520_re_imputation_on_B2018/data/cCRE_document_frequency.npy'
    cCRE_document_frequency = np.load(cCRE_frequency_path)

    # Apply TF-IDF transformation
    adata = global_TFIDF(adata, cCRE_document_frequency)

    # Perform tokenization to create cell sentences
    tokenization(adata)

    # 2. Dataset and Dataloader
    # Extract cell sentences from the AnnData object
    print("Creating Dataset and Dataloader...")
    cell_sentences = adata.obs['cell_sentences'].tolist()

    # Create the training dataset
    train_cell_dataset = CellDatasetForDI(
        adata=adata,
        cell_sentences=cell_sentences,
        max_length=8192,
        alpha_for_CCA=1,
        num_cCRE=1355445,
        is_random=True
    )

    # Create the training DataLoader
    train_batch_size = 8
    train_dataloader = DataLoader(
        train_cell_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=16,
        collate_fn=collate_fn_for_DI
    )

    # Create the inference dataset
    inference_cell_dataset = CellDataset(
        cell_sentences=cell_sentences,
        max_length=8192,
        is_random=True
    )

    # Create the inference DataLoader
    inference_batch_size = 64
    inference_dataloader = DataLoader(
        inference_cell_dataset,
        batch_size=inference_batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # 3. Model
    # Specify the path to the pre-trained model
    model_path = '/user/work/likeyi/EpiAgent/20260520_re_imputation_on_B2018/data/pretrained_EpiAgent.pth'
    print(f"Loading the pre-trained model from path: {model_path}...")

    # Set the device (GPU if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the EpiAgent model with appropriate configurations
    pretrained_model = EpiAgent(
        vocab_size=1355449,
        num_layers=18,
        embedding_dim=512,
        num_attention_heads=8,
        max_rank_embeddings=8192,
        use_flash_attn=True,
        pos_weight_for_RLM=torch.tensor(1.),
        pos_weight_for_CCA=torch.tensor(1.)
    )

    # Set criterion for signal reconstruction (SR)
    pretrained_model.criterion_SR = nn.MSELoss()

    # Load the pre-trained weights into the model
    pretrained_model.load_state_dict(torch.load(model_path))

    # Move the model to the specified device
    pretrained_model.to(device)

    # Fine-tune the model
    print("Fine-tuning the model...")
    fine_tuned_model = fine_tune_epiagent_for_UFE(
        model=pretrained_model,
        train_dataloader=train_dataloader,
        num_steps=50000, 
        save_dir='/user/work/likeyi/EpiAgent/20260520_re_imputation_on_B2018/ckpt_0523/',
        device=device,
        learning_rate=5e-5,
        save_steps=10000,
        log_steps=100,
        warmup_steps=500,
        is_logging=True
    )
    # fine_tuned_model = pretrained_model  # Use the pre-trained model directly without fine-tuning for this run.
    print("Finished!")

    # 4. Inference
    # Call the inference function using the fine-tuned model.
    # We pass the top_50000_indices to restrict the predictions to these cCREs.
    outputs = infer_reconstructed_signals(
        model=fine_tuned_model,
        device=device,
        dataloader=inference_dataloader,
        need_cell_embeddings=False,         # Set to True if you also need cell embeddings.
    )

    predicted_signals = outputs['predicted_signals']
    adata_imputed = adata.copy()
    adata_imputed.X = predicted_signals

    # Save the imputed adata
    output_path = '/user/work/likeyi/EpiAgent/20260520_re_imputation_on_B2018/data/adata_imputed.h5ad'
    adata_imputed.write(output_path)
    print(f"Saved the imputed adata to: {output_path}")

    # 5. Evaluate
    print("Evaluating...")
    evaluation_pipeline(
        dataset_name='Buenrostro2018', 
        raw_h5ad_path='/user/work/likeyi/EpiAgent/20260520_re_imputation_on_B2018/data/Buenrostro2018-bone_marrow_tissue-cell_by_cCRE.h5ad', 
        imputed_h5ad_path=output_path, 
        open_threshold=0.03, 
        celltype_key='Cell_type (HSC)', 
        output_path='/user/work/likeyi/EpiAgent/20260520_re_imputation_on_B2018/result_0523',
        reference_csv_path=None, 
        plot=True
    )

main()
