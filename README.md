# Query-ML: Statistical LLM Query-Based Decision Tree Induction

Active hypothesis generation and interpretable decision-tree induction using Large Language Models (LLMs).

---

## Repository Structure

```
Query-ML/
├── run.py                 # CLI experiment orchestrator & multi-seed evaluation
├── run_experiment.slurm   # SLURM batch job script for GPU cluster execution
├── requirements.txt       # Python dependencies
├── src/
│   ├── __init__.py        # Package exports
│   ├── dataset.py         # Dataset loaders, text formatters, and ground-truth rules
│   ├── evaluate.py        # Evaluation metrics (Accuracy, ROC-AUC, Causal Recovery)
│   ├── generator.py       # vLLM inference engine, batch prompting, and caching
│   ├── prompts.py         # Domain-agnostic query generation and verification prompts
│   ├── reliability.py     # Self-consistency and query verification tests
│   └── tree.py            # Decision tree induction, Shannon entropy, and visualization
├── scripts/
│   ├── slurm/             # Additional SLURM execution scripts
│   └── analysis/          # Result aggregation and plotting utilities
├── data/                  # Benchmark datasets (Admissions, Deceptive Reviews)
└── docs/                  # Research paper and reference notes
```

---

## Getting Started

### 1. Installation (Standalone / Local Machine)

```bash
git clone https://github.com/gupt1156/Query-ML.git
cd Query-ML

conda create -n query-ml python=3.11 -y
conda activate query-ml
pip install -r requirements.txt
```

### 2. Running Local Experiments

Run the contextual benchmark across seeds using local or global regimes:

```bash
# Contextual Admissions Benchmark
python run.py --task_type contextual --seeds 42,52,62 --regimes local global

# Original Admissions Benchmark
python run.py --task_type original --seeds 42,52,62 --min_ig 1e-5
```

---

## HPC & Cluster Execution (SLURM)

To submit array jobs to an HPC cluster (e.g. NYU Torch / Greene with NVIDIA L40S):

```bash
sbatch run_experiment.slurm
```

Logs and checkpoints will be output to `logs/` and `results/`.

### Zero-Configuration Cluster Setup

The execution script (`run_experiment.slurm` and `scripts/slurm/run_experiment.slurm`) is designed to run **out-of-the-box with zero manual setup** for NYU cluster users:

1. **Dynamic Path Detection & Fallbacks**:
   - **`TMPDIR`**: Automatically uses `/scratch/${USER}/tmp` (created dynamically if it does not already exist).
   - **Conda Environment**: Automatically checks if the user has an environment at `/scratch/${USER}/.conda/envs/query-ml` or `~/.conda/envs/query-ml`. If neither is found, it automatically falls back to the shared environment at `/scratch/rvg9413/.conda/envs/query-ml`.
   - **Hugging Face Model Cache**: Checks for pre-cached Qwen-2.5-7B-Instruct weights at `/scratch/rvg9413/.cache/huggingface/...`. If found, it runs fully offline without requiring collaborators to re-download 15 GB of weights. If run on external clusters, it falls back to standard Hugging Face Hub downloads.
   - **Working Directory**: Automatically executes in `${SLURM_SUBMIT_DIR:-.}`, ensuring logs and checkpoints are written to whichever directory the user submitted the job from.
   - **Partition & Allocation**: Uses `--partition=l40s_public` without hardcoded accounts, allowing any authorized user to run immediately under their default SLURM allocation.

2. **Cluster Permissions**:
   - Shared directories on `/scratch/rvg9413` are configured with world-traversal permissions (`chmod o+x`) and NFSv4 ACLs (`A::EVERYONE@:rxtncy`), enabling collaborator read access to the pre-built environment and model cache without permission conflicts.
