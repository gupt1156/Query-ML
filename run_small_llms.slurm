#!/bin/bash
#SBATCH --job-name=qml_small_llms
#SBATCH --account=torch_pr_599_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --partition=h200_tandon,h200_public,l40s_public
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=1-6%6

cd "${SLURM_SUBMIT_DIR:-.}"

SCRATCH_DIR="/scratch/${USER}"
export TMPDIR="${SCRATCH_DIR}/tmp"
export HF_HOME="${SCRATCH_DIR}/.cache/huggingface"
mkdir -p logs results/small_llm_eval "${TMPDIR}" "${HF_HOME}"

PYTHON_BIN="/scratch/${USER}/.conda/envs/query-ml/bin/python"

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)
TAGS=(
    "qwen25_0_5b"
    "tinyllama_1_1b"
    "qwen25_1_5b"
    "smollm2_1_7b"
    "qwen25_3b"
    "qwen25_7b"
)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
MODEL="${MODELS[$IDX]}"
TAG="${TAGS[$IDX]}"

echo "Evaluating ${MODEL} (${TAG}) on $(hostname)..."

${PYTHON_BIN} -u evaluate_small_llms.py \
    --model_name "${MODEL}" \
    --tree_json_pattern "results/vllm_regimes_maj5_final_contextual_seed*/full_experiment_data.json" \
    --output_file "results/small_llm_eval/eval_${TAG}.json"
