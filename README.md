# Query-ML: Statistical LLM Query-Based Decision Tree Induction

Query-ML implements active hypothesis generation and interpretable decision-tree induction using Large Language Models (LLMs). It empirically compares two fundamental statistical learning regimes:

1. **Global Regime (Static Feature Bank):**
   - Samples $M$ contrastive subsets across the entire unpartitioned dataset upfront.
   - Generates a fixed candidate query pool $Q = \{q_1, \dots, q_K\}$.
   - Evaluates all candidate queries across records to build a global binary feature matrix.
   - Fits a standard CART Decision Tree classifier on the resulting static table.
   - **Vulnerability:** Highly susceptible to distractor features that exhibit spurious global correlations, failing to capture subtle local subgroup interactions.

2. **Local Regime (Adaptive Subpopulation Conditioning):**
   - Dynamically proposes targeted, contrastive yes/no queries at each individual tree node.
   - Queries are formulated strictly by contrastively sampling records that have reached the active partition ($D_v$).
   - Evaluates Shannon Information Gain ($IG$) locally to identify the split that maximizes subgroup entropy reduction.
   - **Advantage:** Preserves subpopulation nuance, reliably recovers multi-track disjunctive rules, and suppresses distractor collapse.

---

## 10-Seed Benchmark Suite Results (NVIDIA L40S Cluster)

Evaluated across 10 random seeds: `[42, 52, 62, 72, 82, 92, 102, 112, 122, 132]`.

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

*(Note: Global regime suffered distractor collapse on Seeds 92 and 132, dropping to ~49% accuracy by selecting spurious community service hours, whereas Local achieved 100.0% accuracy on all 10 seeds).*

---

## Codebase Architecture

```
Query-ML/
├── run.py                 # Multi-seed CLI experiment orchestrator & serialization
├── run_parallel.slurm     # Slurm array batch execution script for cluster nodes
├── src/
│   ├── __init__.py        # Public package API exports
│   ├── dataset.py         # Synthetic dataset loader, text formatter, and causal rules
│   ├── evaluate.py        # Accuracy, ROC-AUC, causal recovery, and depth precision
│   ├── generator.py       # vLLM inference engine, contrastive sampling, and caching
│   ├── prompts.py         # 7-rule domain-agnostic meta-prompts & verification templates
│   └── tree.py            # Recursive inductive decision tree, entropy, and Mermaid export
└── data/
    └── admissions/        # Synthetic admissions benchmark dataset files
```

---

## Quickstart

### Environment Setup
```bash
conda create -n query-ml python=3.11 -y
conda activate query-ml
pip install vllm pandas numpy networkx scikit-learn matplotlib
```

### Running Experiments
```bash
# Run Contextual Admissions across 10 seeds
python run.py --task_type contextual --seeds 42,52,62,72,82,92,102,112,122,132

# Run Original Admissions across 10 seeds
python run.py --task_type original --seeds 42,52,62,72,82,92,102,112,122,132 --min_ig 1e-5
```
