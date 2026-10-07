"""
Evaluation Metrics, Causal Recovery Tracking, and Depth Precision Analysis.

This module computes:
1. Standard statistical classification metrics: Test Accuracy, ROC-AUC, Precision, Recall, F1.
2. Causal discovery metrics: Proportion of ground-truth causal mechanisms recovered.
3. Distractor split rates: Proportion of inducted tree splits based on non-causal attributes.
4. Per-depth causal split precision: Precision of query splits at each depth level (0, 1, 2).
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.dataset import is_causal


def compute_classification_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Computes standard statistical classification metrics.
    Safely handles single-class edge cases and missing probabilities for ROC-AUC.
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # ROC-AUC calculation requires both classes to be present in y_true
    has_both_classes = len(np.unique(y_true)) > 1
    if y_prob is not None and has_both_classes:
        auc_score = float(roc_auc_score(y_true, y_prob))
    else:
        auc_score = accuracy

    return {
        "accuracy": accuracy,
        "roc_auc": auc_score,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def aggregate_splits_by_tree_depth(
    splits: List[Dict[str, Any]],
    task_type: str = "contextual"
) -> Dict[int, Dict[str, Any]]:
    """
    Aggregates causal vs distractor split counts across individual tree depth levels (0, 1, 2).
    """
    depth_breakdown: Dict[int, Dict[str, Any]] = {}

    for split_info in splits:
        depth_level = split_info.get("depth", 0)
        category = split_info.get("category", "")
        split_is_causal = is_causal(category, task_type=task_type)

        if depth_level not in depth_breakdown:
            depth_breakdown[depth_level] = {
                "total": 0,
                "causal": 0,
                "distractor": 0,
                "causal_pct": 0.0,
                "distractor_pct": 0.0
            }

        depth_breakdown[depth_level]["total"] += 1
        if split_is_causal:
            depth_breakdown[depth_level]["causal"] += 1
        else:
            depth_breakdown[depth_level]["distractor"] += 1

    # Compute percentage metrics per depth level
    for depth_level, counts in depth_breakdown.items():
        total_at_depth = counts["total"]
        if total_at_depth > 0:
            counts["causal_pct"] = round((counts["causal"] / total_at_depth) * 100, 2)
            counts["distractor_pct"] = round((counts["distractor"] / total_at_depth) * 100, 2)

    return depth_breakdown


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    splits: Optional[List[Dict[str, Any]]] = None,
    task_type: str = "contextual"
) -> Dict[str, Any]:
    """
    Computes the complete evaluation dictionary including accuracy, ROC-AUC,
    causal recovery rate, and distractor split proportions.
    """
    splits_list = splits or []
    total_splits = len(splits_list)

    # 1. Classification performance
    scores = compute_classification_scores(y_true, y_pred, y_prob)

    # 2. Causal mechanism analysis
    causal_splits = sum(1 for s in splits_list if is_causal(s.get("category", ""), task_type))
    distractor_splits = total_splits - causal_splits

    causal_categories_found = {
        s["category"]
        for s in splits_list
        if is_causal(s.get("category", ""), task_type)
    }

    # Ground truth causal feature inventory
    if task_type == "original":
        ground_truth_features = {"true_causal_math"}
    else:
        ground_truth_features = {
            "true_causal_math",
            "true_causal_publications",
            "true_causal_activities"
        }

    # Percentage of true underlying causal features recovered by the inducted tree
    causal_recovery_rate = round(
        (len(causal_categories_found) / len(ground_truth_features)) * 100, 2
    )

    causal_split_pct = round((causal_splits / total_splits) * 100, 2) if total_splits > 0 else 0.0
    distractor_split_pct = round((distractor_splits / total_splits) * 100, 2) if total_splits > 0 else 0.0

    # 3. Per-depth breakdown
    splits_by_depth = aggregate_splits_by_tree_depth(splits_list, task_type=task_type)

    return {
        "accuracy": scores["accuracy"],
        "roc_auc": scores["roc_auc"],
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f1": scores["f1"],
        "total_splits": total_splits,
        "causal_splits": causal_splits,
        "distractor_splits": distractor_splits,
        "causal_split_pct": causal_split_pct,
        "distractor_split_pct": distractor_split_pct,
        "causal_recovery_rate": causal_recovery_rate,
        "causal_features_found": sorted(list(causal_categories_found)),
        "splits_by_depth": splits_by_depth,
        "splits": splits_list
    }
