# EpiAgent

Although single-cell assay for transposase-accessible chromatin using sequencing (scATAC-seq) enables the exploration of the epigenomic landscape that governs transcription at the cellular level, the complicated characteristics of the sequencing data and the broad scope of downstream tasks mean that a sophisticated and versatile computational method is urgently needed. Here, we introduce EpiAgent, a foundation model pretrained on our manually curated large-scale [Human-scATAC-Corpus](https://health.tsinghua.edu.cn/human-scatac-corpus/index.php). EpiAgent encodes chromatin accessibility patterns of cells as concise ‘cell sentences’ and captures cellular heterogeneity behind regulatory networks via bidirectional attention. Comprehensive benchmarks show that EpiAgent excels in typical downstream tasks, including unsupervised feature extraction, supervised cell type annotation and data imputation. By incorporating external embeddings, EpiAgent enables effective cellular response prediction for both out-of-sample stimulated and unseen genetic perturbations, reference data integration and query data mapping. Through in silico knockout of cis-regulatory elements, EpiAgent demonstrates the potential to model cell state changes. EpiAgent is further extended to directly annotate cell types in a zero-shot manner.


<p align="center">
  <img src="https://github.com/xy-chen16/EpiAgent/blob/main/inst/model.png" width="700" height="385" alt="image">
</p>

---

## Updates / News

- **2024.12.21**: Our paper was published on bioRxiv. Read the preprint [here](https://www.biorxiv.org/content/10.1101/2024.12.19.629312v1).
- **2024.12.27**: Source code and Python package released on PyPI under the name `epiagent` (v0.0.1). Install it via `pip install epiagent`.
- **2024.12.28**: Updated GitHub repository with pretrained EpiAgent model and two supervised models for cell type annotation: EpiAgent-B and EpiAgent-NT. Models and example datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1WlNykSCNtZGsUp2oG0dw3cDdVKYDR-iX?usp=sharing). Additionally, we added usage demos for zero-shot applications ([link](https://github.com/xy-chen16/EpiAgent/tree/main/demo/)).
- **2025.02.12**: Updated the `epiagent` PyPI package to version **0.0.2**, adding fine-tuning code for unsupervised feature extraction and supervised cell type annotation. We also provided demos of the fine-tuning code, available [here](https://github.com/xy-chen16/EpiAgent/tree/main/demo/).
- **2025.03.03**: Updated the `epiagent` PyPI package to version **0.0.3**. This release includes new fine-tuning code for: a) data imputation, b) reference data integration and query data mapping, and c) cellular response prediction of out-of-sample stimulated perturbation. In addition, several bugs in the previous version have been fixed. Demo notebooks for fine-tuning EpiAgent for data imputation and for reference data integration and query data mapping are available [here](https://github.com/xy-chen16/EpiAgent/tree/main/demo/).
- **2025.05.22**: Demo notebooks for fine-tuning EpiAgent for perturbation prediction and for in-silico cCRE KO are available [here](https://github.com/xy-chen16/EpiAgent/tree/main/demo/).
- **2025.08.09**: **EpiAgent** has been officially accepted for publication in Nature Methods! 🎉
- **2025.09.09**: Released the full database used for pretraining and downstream applications as the ensemble resource Human-scATAC-Corpus, comprising >5.4 million cells across 37 tissue or cell lines. The database is publicly available at [health.tsinghua.edu.cn/human-scatac-corpus](https://health.tsinghua.edu.cn/human-scatac-corpus).

---

## Installation

### Environment Setup

EpiAgent is built on the **PyTorch 2.0** framework with **FlashAttention v2**. We recommend using **CUDA 11.7** for optimal performance.

#### Step 1: Set up a Python environment

We recommend creating a virtual Python environment with [Anaconda](https://docs.anaconda.com/free/anaconda/install/linux/):

```bash
$ conda create -n EpiAgent python=3.11
$ conda activate EpiAgent
```
#### Step 2: Install Pytorch

Install PyTorch based on your system configuration. Refer to [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/) for the exact command. For example:

```bash
$ pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 # torch 2.0.1 + cuda 11.7
```

#### Step 3: Install FlashAttention (if not already installed)

Install `flash-attn` by following the instructions below (adapted from the [FlashAttention GitHub repository](https://github.com/Dao-AILab/flash-attention/tree/v2.7.2)):

1. FlashAttention uses ninja to compile its C++/CUDA components efficiently. Check if ninja is already installed and working correctly:、:

```bash
$ ninja --version
$ echo $?
```

If the above commands return a nonzero exit code or you encounter errors, reinstall `ninja` to ensure it works properly:

```bash
$ pip uninstall -y ninja && pip install ninja
```

2. Install FlashAttention:

After ensuring ninja is installed, proceed with the `FlashAttention` installation. Use the following command to install a compatible version:

```bash
$ pip install flash-attn==2.5.8 --no-build-isolation
```

#### Step 4: Install EpiAgent and dependencies

To install EpiAgent, run:

```bash
$ pip install epiagent
```

## Data Preprocessing

EpiAgent uses a unified set of **candidate cis-regulatory elements (cCREs)** as features. We recommend starting from fragment files to process input data compatible with EpiAgent. The preprocessing steps include:

1. **Reference Genome Conversion (Optional):**
   - Our cCRE coordinates are based on hg38. If your fragment files use hg19, use `liftOver` to convert them to hg38.

2. **Fragment Overlap Calculation:**
   - Use `bedtools` to calculate overlaps between fragments and cCREs.

3. **Cell-by-cCRE Matrix Construction:**
   - Use `epiagent.preprocessing.construct_cell_by_ccre_matrix` to create the cell-by-cCRE matrix and add metadata.

4. **TF-IDF and Tokenization:**
   - Perform global TF-IDF to assign importance to accessible cCREs, followed by tokenization to generate cell sentences.

For a detailed example, refer to the demo notebook: [Data Preprocessing.ipynb](https://github.com/xy-chen16/EpiAgent/blob/main/demo/Data%20Preprocessing.ipynb).

---

## Downstream Analysis

### Zero-shot unsupervised feature extraction with the pretrained EpiAgent model

- Pretrained EpiAgent model parameters and example files are available [here](https://drive.google.com/drive/folders/1WlNykSCNtZGsUp2oG0dw3cDdVKYDR-iX?usp=sharing).
- A demo for zero-shot feature extraction is available in [Zero-shot Feature Extraction using EpiAgent.ipynb](https://github.com/xy-chen16/EpiAgent/blob/main/demo/Zero-shot%20Feature%20Extraction%20using%20EpiAgent.ipynb).

### Fine-tuning EpiAgent for unsupervised feature extraction

- Pretrained EpiAgent model parameters and example files are available [here](https://drive.google.com/drive/folders/1WlNykSCNtZGsUp2oG0dw3cDdVKYDR-iX?usp=sharing).
- A demo for fine-tuning EpiAgent for unsupervised feature extraction is available in [Fine-tuning EpiAgent for Unsupervised Feature Extraction.ipynb](https://github.com/xy-chen16/EpiAgent/blob/main/demo/Fine-tuning%20EpiAgent%20for%20unsupervised%20feature%20extraction.ipynb).

### Fine-tuning EpiAgent for supervised cell type annotation

- Pretrained EpiAgent model parameters and example files are available [here](https://drive.google.com/drive/folders/1WlNykSCNtZGsUp2oG0dw3cDdVKYDR-iX?usp=sharing).
- A demo for fine-tuning EpiAgent for supervised cell type annotation is available in [Fine-tuning EpiAgent for Supervised Cell Type Annotation.ipynb](https://github.com/xy-chen16/EpiAgent/blob/main/demo/Fine-tuning%20EpiAgent%20for%20supervised%20cell%20type%20annotation.ipynb).

### Data Imputation

- Pretrained EpiAgent model parameters and example files are available [here](https://drive.google.com/drive/folders/1WlNykSCNtZGsUp2oG0dw3cDdVKYDR-iX?usp=sharing).
- A demo for fine-tuning EpiAgent for data imputation is available in [Fine-tuning EpiAgent for Data Imputation.ipynb](https://github.com/xy-chen16/EpiAgent/blob/main/demo/Fine-tuning%20EpiAgent%20for%20data%20imputation.ipynb).

### Reference Data Integration and Query Data Mapping

- Pretrained EpiAgent model parameters and example files are available [here](https://drive.google.com/drive/folders/1WlNykSCNtZGsUp2oG0dw3cDdVKYDR-iX?usp=sharing).
- A demo for fine-tuning EpiAgent for reference data integration and query data mapping is available in [Fine-tuning EpiAgent for Reference Data Integration and Query Data Mapping.ipynb](https://github.com/xy-chen16/EpiAgent/blob/main/demo/Fine-tuning%20EpiAgent%20for%20reference%20data%20integration%20and%20query%20data%20mapping.ipynb).

### Zero-shot cell type annotation with EpiAgent-B and EpiAgent-NT

Two supervised models, **EpiAgent-B** and **EpiAgent-NT**, are designed for direct cell type annotation. These models and their example datasets can be downloaded [here](https://drive.google.com/drive/folders/1WlNykSCNtZGsUp2oG0dw3cDdVKYDR-iX?usp=sharing). For specific demos:

- Annotating brain cell datasets with **EpiAgent-B**: [Zero-shot annotation using EpiAgent-B.ipynb](https://github.com/xy-chen16/EpiAgent/blob/main/demo/Zero-shot%20annotation%20using%20EpiAgent-B.ipynb)
- Annotating other tissue datasets with **EpiAgent-NT**: [Zero-shot annotation using EpiAgent-NT.ipynb](https://github.com/xy-chen16/EpiAgent/blob/main/demo/Zero-shot%20annotation%20using%20EpiAgent-NT.ipynb)

### Prediction of Cellular Responses to Perturbations

- Pretrained EpiAgent model parameters and example files are available [here](https://drive.google.com/drive/folders/1WlNykSCNtZGsUp2oG0dw3cDdVKYDR-iX?usp=sharing).
- A demo for fine-tuning EpiAgent for prediction of cellular responses to perturbations is available in: [Fine-tuning EpiAgent for perturbation prediction.ipynb](https://github.com/xy-chen16/EpiAgent/blob/main/demo/Fine-tuning%20EpiAgent%20for%20perturbation%20prediction.ipynb)

### In-silico cCRE KO
- A demo for fine-tuning EpiAgent for in-silico cCRE KO is available in [In-silico cCRE KO.ipynb](https://github.com/xy-chen16/EpiAgent/blob/main/demo/In-silico%20cCRE%20KO.ipynb)


---

## Citation

If you use EpiAgent in your research, please cite our paper:

Chen X, Li K, Cui X, Wang Z, Jiang Q, Lin J, Li Z, Gao Z, Jiang R. EpiAgent: Foundation model for single-cell epigenomic data. bioRxiv. 2024:2024-12.

---

## Contact

For questions about the paper or code, please email: xychen20@mails.tsinghua.edu.cn

