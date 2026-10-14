# Query-ML: Statistical LLM Query-Based Decision Tree Induction

Active hypothesis generation and interpretable decision-tree induction using Large Language Models (LLMs).

---

## Repository Structure

| File / Directory | Description |
| :--- | :--- |
| `run.py` | Multi-seed inductive decision tree runner (`QueryDecisionTree`) |
| `evaluate_small_llms.py` | Benchmark suite for sub-$7\text{B}$ and $7\text{B}$ LLMs (Direct vs. Tree-Guided, Simplicity Audit, Speedups) |
| `generate_pre_meeting_figures.py` | Summary JSON and publication-quality figure generator |
| `run_experiment.slurm` | SLURM GPU array job script for 50-seed tree induction |
| `run_small_llms.slurm` | SLURM GPU array job script for evaluating small LLMs |
| `src/__init__.py` | Package exports |
| `src/generator.py` | Domain-agnostic prompt builders, `FastVLLMEngine`, batching, and vote caching |
| `src/tree.py` | Inductive decision tree (`QueryDecisionTree`), consistency gating, and bottom-up pruning |
| `src/utils.py` | Semantic query parsing, sanitization, vote consistency math, dataset loader, and metrics |
| `data/` | Benchmark datasets (`admission_train.json`, `admission_val.json`, `admission_test.json`) |

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

### 2. Running Experiments

```bash
# Multi-seed tree induction (5-vote majority voting)
python run.py --seeds 42,52,62 --majority_votes 5 --output_dir results/vllm_regimes_maj5_final

# Evaluate a small LLM (Direct vs. Tree-Guided, Query Audit, and Speedups)
python evaluate_small_llms.py --model_name Qwen/Qwen2.5-1.5B-Instruct --output_file results/small_llm_eval/eval_qwen25_1_5b.json

# Generate summary statistics and figures
python generate_pre_meeting_figures.py
```

---

## HPC & Cluster Execution (SLURM)

```bash
# 50-seed tree induction across 5 GPU array tasks
sbatch run_experiment.slurm

# Benchmark 5 small LLMs + 7B reference across 6 GPU array tasks
sbatch run_small_llms.slurm
```
