#!/bin/bash
#SBATCH --job-name=queries_admissions_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:h100:4
#SBATCH --time=01:00:00
#SBATCH --partition=h100_tandon
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ic2664@nyu.edu
#SBATCH --account=torch_pr_599_tandon_advanced

# ── Environment ──────────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.06
source activate query-ml
export PATH="/scratch/ic2664/.conda/envs/query-ml/bin:$PATH"

# ── Working directory ─────────────────────────────────────────────────────────
cd $SLURM_SUBMIT_DIR

# ── Logging ───────────────────────────────────────────────────────────────────
mkdir -p logs
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "GPUs:       $CUDA_VISIBLE_DEVICES"
nvidia-smi

# ── Run ───────────────────────────────────────────────────────────────────────
START_RUN=${START_RUN:-1}
START_RUN=$START_RUN /scratch/ic2664/.conda/envs/query-ml/bin/python queries_ML_admissions_test.py

echo "End time: $(date)"
