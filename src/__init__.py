"""
Query-ML: Interpretable Statistical Machine Learning via LLM Queries.

Exports:
- QueryDecisionTree: Inductive decision tree supporting Local and Global regimes.
- FastVLLMEngine: High-throughput vLLM query generation and cached answer engine.
- load_admissions_data: Dataset loader for synthetic admissions benchmarks.
- compute_metrics: Comprehensive classification and causal recovery evaluation.
"""

from src.dataset import load_admissions_data
from src.evaluate import compute_metrics
from src.generator import FastVLLMEngine
from src.tree import QueryDecisionTree

__version__ = "0.2.0"
__all__ = [
    "QueryDecisionTree",
    "FastVLLMEngine",
    "load_admissions_data",
    "compute_metrics",
    "__version__"
]
