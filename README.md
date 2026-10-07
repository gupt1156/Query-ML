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

### 1. Installation

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

### 3. Running on SLURM Cluster

To submit array jobs to an HPC cluster (e.g. NYU Torch / Greene with NVIDIA L40S):

```bash
sbatch run_experiment.slurm
```

Logs and checkpoints will be output to `logs/` and `results/`.
