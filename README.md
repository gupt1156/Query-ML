# Query-ML: Statistical LLM Query-Based Decision Tree Induction

Active hypothesis generation and interpretable decision-tree induction using Large Language Models (LLMs).

---

## Repository Structure

```
Query-ML/
├── run.py                 # CLI experiment orchestrator & multi-seed evaluation
├── run_experiment.slurm   # SLURM batch job script for GPU cluster execution
├── environment.yml        # Conda environment specification
├── requirements.txt       # Python package dependencies
├── src/
│   ├── __init__.py        # Package exports
│   ├── dataset.py         # Dataset loaders, text formatters, and ground-truth rules
│   ├── evaluate.py        # Evaluation metrics (Accuracy, ROC-AUC, Causal Recovery)
│   ├── generator.py       # vLLM inference engine, batch prompting, and caching
│   ├── prompts.py         # Domain-agnostic query generation and verification prompts
│   ├── reliability.py     # Self-consistency and query verification tests
│   └── tree.py            # Decision tree induction, Shannon entropy, and visualization
├── data/                  # Benchmark datasets (Admissions, Deceptive Reviews)
└── docs/                  # Research paper and reference notes
```

---

## Getting Started

### 1. Environment Setup

Clone the repository and set up the Python environment using either Conda or Pip:

#### Option A: Using Conda (Recommended)
```bash
git clone https://github.com/gupt1156/Query-ML.git
cd Query-ML

conda env create -f environment.yml
conda activate query-ml
```

#### Option B: Using Pip
```bash
git clone https://github.com/gupt1156/Query-ML.git
cd Query-ML

conda create -n query-ml python=3.11 -y
conda activate query-ml
pip install -r requirements.txt
```

### 2. Model Weights & Cache

By default, Query-ML uses [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).
- On the first run, vLLM will automatically download the model weights (~15GB) into your Hugging Face cache directory (`$HF_HOME`, defaulting to `~/.cache/huggingface` or `/scratch/${USER}/.cache/huggingface`).
- If you already have model weights downloaded locally, you can specify them via `--model_name /path/to/model/checkpoint`.

### 3. Running Local Experiments

Run the contextual benchmark across seeds:

```bash
# Contextual Admissions Benchmark
python run.py --task_type contextual --seeds 42,52,62 --regimes local global

# Original Admissions Benchmark
python run.py --task_type original --seeds 42,52,62 --min_ig 1e-5
```

---

## HPC & Cluster Execution (SLURM)

To submit array jobs to an HPC cluster (e.g., NYU Torch / Greene with NVIDIA L40S GPUs):

```bash
sbatch run_experiment.slurm
```

Logs will be saved to `logs/` and experiment checkpoints to `results/`.

### Fully Portable & Autonomous Cluster Setup

`run_experiment.slurm` is fully self-contained and requires **no hardcoded paths or shared scratch access**:
- **Automatic Scratch Directory**: Sets `TMPDIR` and `HF_HOME` to `/scratch/${USER}` (or `${SLURM_TMPDIR}` / `$HOME/scratch` if running on non-NYU clusters).
- **Environment Auto-Activation**: Automatically locates and activates the `query-ml` conda environment from your user environment.
- **Model Download & Caching**: Downloads and caches `Qwen/Qwen2.5-7B-Instruct` into the running user's scratch directory on first execution, requiring zero access to any other user's files.
- **Custom Models**: You can override the default model at launch time:
  ```bash
  MODEL_NAME="/path/to/custom/weights" sbatch run_experiment.slurm
  ```
