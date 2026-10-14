#!/bin/bash
#SBATCH --job-name=qml_experiment
#SBATCH --account=torch_pr_599_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --partition=h200_tandon,h200_public,l40s_public
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=1-5%5

# -----------------------------------------------------------------------------
# Configuration (Self-contained and portable for any user)
# -----------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-.}"

# Temporary scratch and model cache directories
if [ -d "/scratch/${USER}" ]; then
    SCRATCH_DIR="/scratch/${USER}"
elif [ -n "${SLURM_TMPDIR}" ] && [ -d "${SLURM_TMPDIR}" ]; then
    SCRATCH_DIR="${SLURM_TMPDIR}"
else
    SCRATCH_DIR="${HOME}/scratch"
fi

export TMPDIR="${SCRATCH_DIR}/tmp"
export HF_HOME="${SCRATCH_DIR}/.cache/huggingface"
mkdir -p logs results "${TMPDIR}" "${HF_HOME}"

# Environment activation:
# Activate conda environment 'query-ml' if conda is available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate query-ml 2>/dev/null || true
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate query-ml 2>/dev/null || true
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    conda activate query-ml 2>/dev/null || true
fi

# Locate python binary
if [ -n "${CONDA_PREFIX}" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif [ -x "${HOME}/.conda/envs/query-ml/bin/python" ]; then
    PYTHON_BIN="${HOME}/.conda/envs/query-ml/bin/python"
elif [ -x "/scratch/${USER}/.conda/envs/query-ml/bin/python" ]; then
    PYTHON_BIN="/scratch/${USER}/.conda/envs/query-ml/bin/python"
else
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi

# Model identifier (defaults to Hugging Face Hub, auto-downloaded on first run)
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
MAJORITY_VOTES="${MAJORITY_VOTES:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-results/vllm_regimes_maj${MAJORITY_VOTES}}"

# vLLM runtime settings
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# -----------------------------------------------------------------------------
# Experiment Execution (50 seeds across 5 array tasks = 10 seeds per GPU job,
# amortizing vLLM model load and cross-seed record vote cache)
# -----------------------------------------------------------------------------
SEED_GROUPS=(
    "42,52,62,72,82,92,102,112,122,132"
    "142,152,162,172,182,192,202,212,222,232"
    "242,252,262,272,282,292,302,312,322,332"
    "342,352,362,372,382,392,402,412,422,432"
    "442,452,462,472,482,492,502,512,522,532"
)
IDX=$((SLURM_ARRAY_TASK_ID - 1))
SEEDS="${SEED_GROUPS[$IDX]}"

echo "Running Query-ML on $(hostname) with seeds: ${SEEDS}..."
echo "Using Python: ${PYTHON_BIN}"
echo "Using Model:  ${MODEL_NAME}"
echo "HF_HOME:      ${HF_HOME}"

${PYTHON_BIN} -u run.py \
    --seeds ${SEEDS} \
    --local_m 6 \
    --k_queries 5 \
    --max_depth 3 \
    --min_ig 0.01 \
    --majority_votes "${MAJORITY_VOTES}" \
    --min_unanimous_consistency 0.80 \
    --min_high_consistency 0.90 \
    --min_error_reduction 2 \
    --model_name "${MODEL_NAME}" \
    --output_dir "${OUTPUT_DIR}"


