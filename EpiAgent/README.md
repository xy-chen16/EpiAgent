# EpiAgent

Large-scale foundation models have recently opened new avenues for artificial general intelligence. Such a research paradigm has recently shown considerable promise in the analysis of single-cell sequencing data, while to date, efforts have centered on transcriptome. In contrast to gene expression, chromatin accessibility provides more decisive insights into cell states, shaping the chromatin regulatory landscapes that control transcription in distinct cell types. Yet, challenges also persist due to the abundance of features, high data sparsity, and the quasi-binary nature of these data. Here, we introduce EpiAgent, the first foundation model for single-cell epigenomic data, pretrained on a large-scale Human-scATAC-Corpus comprising approximately 5 million cells and 35 billion tokens. EpiAgent encodes chromatin accessibility patterns of cells as concise “cell sentences,” and employs bidirectional attention to capture cellular heterogeneity behind regulatory networks. With comprehensive benchmarks, we demonstrate that EpiAgent excels in typical downstream tasks, including unsupervised feature extraction, supervised cell annotation, and data imputation. By incorporating external embeddings, EpiAgent facilitates the prediction of cellular responses to both out-of-sample stimulated and unseen genetic perturbations, as well as reference data integration and query data mapping. By simulating the knockout of key cis-regulatory elements, EpiAgent enables in-silico treatment for cancer analysis. We further extended zero-shot capabilities of EpiAgent, allowing direct cell type annotation on newly sequenced datasets without additional training.

<p align="center">
  <img src="https://github.com/xy-chen16/EpiAgent/blob/main/inst/model.png" width="700" height="385" alt="image">
</p>

---

## Updates / News

- **2024.12.21**: Our paper was published on bioRxiv. Read the preprint [here](https://www.biorxiv.org/content/10.1101/2024.12.19.629312v1).
- **2024.12.27**: Source code and Python package released on PyPI under the name `epiagent` (v0.0.1). Install it via `pip install epiagent`.
- **2024.12.28**: Updated GitHub repository with pretrained EpiAgent model and two supervised models for cell type annotation: EpiAgent-B and EpiAgent-NT. Models and example datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1WlNykSCNtZGsUp2oG0dw3cDdVKYDR-iX?usp=sharing). Additionally, we added usage demos for zero-shot applications ([link](https://github.com/xy-chen16/EpiAgent/demo/)).

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

For a detailed example, refer to the demo notebook: [Data Preprocessing.ipynb](https://github.com/xy-chen16/EpiAgent/demo/Data%20Preprocessing.ipynb).

---

## Downstream Analysis

### Feature Extraction
- Pretrained EpiAgent model parameters and example files are available [here](https://drive.google.com/drive/folders/1WlNykSCNtZGsUp2oG0dw3cDdVKYDR-iX?usp=sharing).
- A demo for zero-shot feature extraction is available in [Zero-shot Feature Extraction using EpiAgent.ipynb](https://github.com/xy-chen16/EpiAgent/demo/Zero-shot%20Feature%20Extraction%20using%20EpiAgent.ipynb).

### Direct Cell Type Annotation

Two supervised models, **EpiAgent-B** and **EpiAgent-NT**, are designed for direct cell type annotation. These models and their example datasets can be downloaded [here](https://drive.google.com/drive/folders/1WlNykSCNtZGsUp2oG0dw3cDdVKYDR-iX?usp=sharing). For specific demos:

- Annotating brain cell datasets with **EpiAgent-B**: [Zero-shot annotation using EpiAgent-B.ipynb](https://github.com/xy-chen16/EpiAgent/demo/Zero-shot%20annotation%20using%20EpiAgent-B.ipynb)
- Annotating other tissue datasets with **EpiAgent-NT**: [Zero-shot annotation using EpiAgent-NT.ipynb](https://github.com/xy-chen16/EpiAgent/demo/Zero-shot%20annotation%20using%20EpiAgent-NT.ipynb)

### Other tasks
- **Data Imputation**
- **Prediction of Cellular Responses to Stimulations and Genetic Perturbations**
- **Reference Data Integration and Query Data Mapping**
- **In-silico Treatment Simulations**

Fine-tuning and additional code demos will be updated soon.

---

## Citation

If you use EpiAgent in your research, please cite our paper:

Chen X, Li K, Cui X, Wang Z, Jiang Q, Lin J, Li Z, Gao Z, Jiang R. EpiAgent: Foundation model for single-cell epigenomic data. bioRxiv. 2024:2024-12.

---

## Contact

For questions about the paper or code, please email: xychen20@mails.tsinghua.edu.cn

