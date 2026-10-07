# Query-ML: Statistical LLM Query-Based Decision Tree Induction

Query-ML implements active hypothesis generation and interpretable decision-tree induction using Large Language Models (LLMs). It compares two statistical learning regimes:

1. **Global Regime (Static Feature Bank):**
   - Samples contrastive subsets across the unpartitioned dataset upfront to generate candidate queries.
   - Evaluates all candidate queries across records to build a global binary feature matrix.
   - Fits a standard CART Decision Tree classifier on the resulting static table.
   - **Vulnerability:** Prone to selecting distractor features that exhibit spurious global correlations, missing fine-grained subpopulation interactions.

2. **Local Regime (Adaptive Subpopulation Conditioning):**
   - Dynamically generates targeted contrastive yes/no queries at each tree node.
   - Queries are formulated using records conditioned on the active partition ($D_v$).
   - Maximizes Shannon Information Gain ($IG$) locally to identify splits that reduce subgroup entropy.
   - **Advantage:** Preserves subpopulation nuance, recovers disjunctive rules, and suppresses distractor collapse.

---

## Benchmark Results (NVIDIA L40S Cluster)

Evaluated across 10 random seeds (`[42, 52, 62, 72, 82, 92, 102, 112, 122, 132]`):

### 1. Contextual Admissions Benchmark (Dual-Track Subgroups)
$$\text{Ground Truth: } \text{Admitted} \iff (\text{Math} = \text{'A'} \land \text{Publications} \ge 1) \lor (\text{Math} \ne \text{'A'} \land \text{Activities} \ge 2)$$

| Metric | Global Regime | Local Regime | Local Advantage ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | $76.40\% \pm 2.81\%$ | **$84.10\% \pm 4.67\%$** | **$+7.70\%$ (100% win rate)** |
| **ROC-AUC** | $0.8300 \pm 0.0350$ | **$0.9054 \pm 0.0446$** | **$+7.54\%$** |
| **Causal Feature Recovery** | $73.3\%$ | **$93.3\%$** | **$+20.0\%$** |
| **Distractor Split Rate** | $22.8\%$ | **$16.7\%$** | **$-6.1\%$ (Lower is better)** |
| **Root (Depth 0) Precision** | $76.9\%$ ($40/52$) | **$100.0\%$ ($10/10$)** | **Zero Root Distractors** |
| **Branch (Depth 1) Precision** | N/A ($0/0$) | **$100.0\%$ ($20/20$)** | **100% Causal Isolation** |

### 2. Original Admissions Benchmark (Single-Track Rule)
$$\text{Ground Truth: } \text{Admitted} \iff \text{Math} = \text{'A'}$$

| Metric | Global Regime | Local Regime | Local Advantage ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | $89.80\% \pm 20.40\%$ | **$100.00\% \pm 0.00\%$** | **$+10.20\%$ (Zero Variance)** |
| **ROC-AUC** | $0.9029 \pm 0.1941$ | **$1.0000 \pm 0.0000$** | **$+0.0971$** |
| **Causal Recovery Rate** | $80.0\%$ | **$100.0\%$** | **$+20.0\%$** |
| **Distractor Split Rate** | $20.0\%$ | **$3.3\%$** | **$-16.7\%$** |

---

## Repository Structure

```
Query-ML/
├── run.py                     # CLI experiment orchestrator & multi-seed evaluation
├── requirements.txt           # Python dependencies
├── src/
│   ├── __init__.py            # Package exports
│   ├── dataset.py             # Dataset loaders, text formatters, and ground-truth rules
│   ├── evaluate.py            # Evaluation metrics (Accuracy, ROC-AUC, Causal Recovery)
│   ├── generator.py           # vLLM inference engine, batch prompting, and caching
│   ├── prompts.py             # Domain-agnostic query generation and verification prompts
│   ├── reliability.py         # Self-consistency and query verification tests
│   └── tree.py                # Decision tree induction, Shannon entropy, and visualization
├── scripts/
│   ├── slurm/
│   │   └── run_experiment.slurm # SLURM job script for GPU cluster execution
│   └── analysis/              # Result aggregation and plotting utilities
├── data/                      # Benchmark datasets (Admissions, Deceptive Reviews)
└── docs/                      # Research paper and reference notes
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
sbatch scripts/slurm/run_experiment.slurm
```

Results and logs will be written to `results/` and `logs/`.
