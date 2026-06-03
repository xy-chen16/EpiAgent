import os
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from epiagent.preprocessing import global_TFIDF
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score,
    adjusted_mutual_info_score,
)

def run_louvain(
    adata,
    label_key,
    cluster_key="louvain",
    range_min=0.0,
    range_max=3.0,
    max_steps=30,
    use_rep=None,
    inplace=True,
    verbose=True,
):

    if "neighbors" not in adata.uns:
        if verbose:
            print("Computing neighbors...")
        sc.pp.neighbors(adata, use_rep=use_rep)

    target_n_clusters = adata.obs[label_key].nunique()

    left = float(range_min)
    right = float(range_max)

    records = []
    best_clustering = None
    found = False

    for step in range(max_steps):
        resolution = left + (right - left) / 2

        sc.tl.louvain(
            adata,
            resolution=resolution,
            key_added=cluster_key,
        )

        current_n_clusters = adata.obs[cluster_key].nunique()

        NMI_score = normalized_mutual_info_score(
            adata.obs[label_key],
            adata.obs[cluster_key],
        )
        ARI_score = adjusted_rand_score(
            adata.obs[label_key],
            adata.obs[cluster_key],
        )
        Homo_score = homogeneity_score(
            adata.obs[label_key],
            adata.obs[cluster_key],
        )
        AMI_score = adjusted_mutual_info_score(
            adata.obs[label_key],
            adata.obs[cluster_key],
        )

        records.append(
            {
                "step": step + 1,
                "resolution": resolution,
                "n_clusters": current_n_clusters,
                "target_n_clusters": target_n_clusters,
                "NMI": NMI_score,
                "ARI": ARI_score,
                "Homo": Homo_score,
                "AMI": AMI_score,
                "matched": current_n_clusters == target_n_clusters,
            }
        )

        if verbose:
            print(
                f"step: {step + 1}, "
                f"resolution: {resolution:.6f}, "
                f"clusters: {current_n_clusters}, "
                f"target: {target_n_clusters}, "
                f"NMI: {NMI_score:.4f}, "
                f"ARI: {ARI_score:.4f}"
            )

        if current_n_clusters > target_n_clusters:
            right = resolution

        elif current_n_clusters < target_n_clusters:
            left = resolution

        else:
            best_clustering = adata.obs[cluster_key].copy()
            found = True
            break

        del adata.obs[cluster_key]

    score_all = pd.DataFrame(records)

    if found:
        if inplace:
            adata.obs[cluster_key] = best_clustering
    else:
        if cluster_key in adata.obs.columns:
            del adata.obs[cluster_key]

        if verbose:
            print(
                "Warning: binary search did not find a resolution with exactly "
                f"{target_n_clusters} clusters within {max_steps} steps."
            )

    return score_all


def clustering_pipeline(raw_h5ad_path, imputed_h5ad_path, open_threshold=0.03, celltype_key='cell_type', output_path='./'):

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

    celltype_key = "Cell_type (HSC)" 

    print("Running PCA...")
    sc.tl.pca(adata, n_comps=50)

    print("Computing neighbor graph...")
    sc.pp.neighbors(adata, n_neighbors=15)

    print("Running Louvain clustering...")
    df = run_louvain(adata,celltype_key,'cluster')
    save_path = os.path.join(output_path, "cluster.csv")
    df.to_csv(save_path)

    print("Running UMAP...")
    sc.tl.umap(adata)

    print("Plotting UMAP...")
    fig, ax = plt.subplots(figsize=(7, 6))

    sc.pl.umap(
        adata,
        color=celltype_key,
        ax=ax,
        show=False,
        title=f"UMAP - Imputed Data ({celltype_key})",
    )

    save_path = os.path.join(output_path, "imputed_adata_umap.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    output_path = "/user/work/likeyi/EpiAgent/20260520_re_imputation_on_B2018/result_0523"
    os.makedirs(output_path, exist_ok=True)
    clustering_pipeline(
        raw_h5ad_path='/user/work/likeyi/EpiAgent/20260520_re_imputation_on_B2018/data/Buenrostro2018-bone_marrow_tissue-cell_by_cCRE.h5ad', 
        imputed_h5ad_path='/user/work/likeyi/EpiAgent/20260520_re_imputation_on_B2018/data/adata_imputed.h5ad', 
        open_threshold=0.03, 
        celltype_key='Cell_type (HSC)', 
        output_path=output_path
    )

main()