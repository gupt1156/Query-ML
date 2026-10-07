#!/bin/bash
#SBATCH --job-name=qml_experiment
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=1-10%5

# -----------------------------------------------------------------------------
# Configuration (Dynamic paths: works for any user out of the box)
# -----------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-.}"

USER_SCRATCH="/scratch/${USER}"
SHARED_SCRATCH="/scratch/rvg9413"

# Set up local directories
export TMPDIR="${USER_SCRATCH}/tmp"
mkdir -p logs results "${TMPDIR}"

# 1. Environment: Use user env if available, otherwise fallback to shared env
if [ -d "${USER_SCRATCH}/.conda/envs/query-ml" ]; then
    CONDA_ENV="${USER_SCRATCH}/.conda/envs/query-ml"
elif [ -d "${HOME}/.conda/envs/query-ml" ]; then
    CONDA_ENV="${HOME}/.conda/envs/query-ml"
else
    CONDA_ENV="${SHARED_SCRATCH}/.conda/envs/query-ml"
fi

export PATH="${CONDA_ENV}/bin:${PATH}"
PYTHON_BIN="${CONDA_ENV}/bin/python"

# 2. Model: Use shared pre-downloaded Qwen model checkpoint if available
SHARED_MODEL="${SHARED_SCRATCH}/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

if [ -d "${SHARED_MODEL}" ]; then
    MODEL_PATH="${SHARED_MODEL}"
    export HF_HOME="${SHARED_SCRATCH}/.cache/huggingface"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
else
    MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
    export HF_HOME="${USER_SCRATCH}/.cache/huggingface"
fi

# 3. vLLM execution settings
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# -----------------------------------------------------------------------------
# Experiment Execution
# -----------------------------------------------------------------------------
SEEDS=(42 52 62 72 82 92 102 112 122 132)
IDX=$((SLURM_ARRAY_TASK_ID - 1))
SEED=${SEEDS[$IDX]}

echo "Running Query-ML on $(hostname) with seed ${SEED}..."
echo "Using Python: ${PYTHON_BIN}"
echo "Using Model:  ${MODEL_PATH}"

${PYTHON_BIN} -u run.py \
    --task_type contextual \
    --seeds ${SEED} \
    --global_M 6 \
    --local_m 6 \
    --k_queries 5 \
    --max_depth 3 \
    --min_ig 0.01 \
    --max_correlation 0.85 \
    --majority_votes 5 \
    --regimes local \
    --model_name "${MODEL_PATH}" \
    --output_dir results/contextual_runs
