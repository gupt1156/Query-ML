#!/bin/bash
#SBATCH --job-name=qml_orig10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=torch_pr_599_tandon_advanced
#SBATCH --array=1-10

module purge
module load anaconda3/2025.06

export TMPDIR=/scratch/rvg9413/tmp
export HF_HOME=/scratch/rvg9413/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PATH=/scratch/rvg9413/.conda/envs/query-ml/bin:$PATH
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p logs results/vllm_regimes $TMPDIR $HF_HOME
cd /scratch/rvg9413/Query-ML

PYTHON_BIN=/scratch/rvg9413/.conda/envs/query-ml/bin/python
MODEL_PATH=/scratch/rvg9413/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28

SEEDS=(42 52 62 72 82 92 102 112 122 132)
IDX=$((SLURM_ARRAY_TASK_ID - 1))
SEED=${SEEDS[$IDX]}
echo "Running Original Admissions Seed $SEED on $SLURMD_NODENAME..."
$PYTHON_BIN -u run.py --task_type original --seeds $SEED --global_M 5 --local_m 8 --k_queries 5 --max_depth 2 --min_ig 1e-5 --model_name $MODEL_PATH
