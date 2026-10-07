#!/bin/bash
#SBATCH --job-name=qml_rel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=torch_pr_599_tandon_advanced

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

mkdir -p logs results/reliability $TMPDIR $HF_HOME
cd /scratch/rvg9413/Query-ML

PYTHON_BIN=/scratch/rvg9413/.conda/envs/query-ml/bin/python
MODEL_PATH=/scratch/rvg9413/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28

echo "Starting Ground-Truth Reliability Benchmark on $SLURMD_NODENAME..."
$PYTHON_BIN -u src/reliability.py \
    --model_name $MODEL_PATH \
    --output_file results/reliability/reliability_benchmark.json \
    --n_runs 5 \
    --temperature 0.7
