#!/bin/bash
#SBATCH --job-name=Imputation
#SBATCH --nodes=1
#SBATCH --output=/user/work/likeyi/EpiAgent/20260520_re_imputation_on_B2018/result_0523/sbatch_log/%x_%j.out

source /home/likeyi/anaconda3/etc/profile.d/conda.sh
conda activate flashattn-v3
nvidia-smi
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u ~/program/EpiAgent/20260520_re_imputation_on_B2018/data_imputation.py