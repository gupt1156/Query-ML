"""
Unified Decision Tree Estimator for Natural Language Queries.

This module provides inductive decision tree learning over text data using LLM queries:
1. Local Regime: Adaptively generates contrastive queries conditioned on local node
   subpopulations (D_v), isolating causal subgroup features without dilution.
2. Global Regime: Generates a static query pool globally upfront and learns a CART tree,
   exposing the failure modes of global pooling under distractor features.
3. Non-Redundant Tree Induction: Eliminates repetitive and contradictory questions
   via branch-level ancestor conditioning, attribute-level uniqueness, and correlation pruning.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.dataset import classify_query, is_causal
from src.evaluate import compute_metrics
from src.generator import FastVLLMEngine

__all__ = ["QueryDecisionTree", "classify_query", "is_causal", "compute_metrics"]


def compute_shannon_entropy(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    p = float(np.mean(labels))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(- p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))


def compute_split_gain(
    labels: np.ndarray,
    binary_mask: np.ndarray,
    parent_entropy: float
) -> Tuple[float, int, int]:
    total_samples = len(labels)
    positive_mask = (binary_mask == 1)
    negative_mask = (binary_mask == 0)

    count_positive = int(positive_mask.sum())
    count_negative = int(negative_mask.sum())

    if count_positive == 0 or count_negative == 0:
        return 0.0, count_positive, count_negative

    entropy_pos = compute_shannon_entropy(labels[positive_mask])
    entropy_neg = compute_shannon_entropy(labels[negative_mask])

    weighted_child_entropy = (
        (count_positive / total_samples) * entropy_pos
        + (count_negative / total_samples) * entropy_neg
    )
    information_gain = parent_entropy - weighted_child_entropy
    return float(information_gain), count_positive, count_negative


def find_optimal_split_query(
    candidate_queries: List[str],
    answers_df: pd.DataFrame,
    labels: np.ndarray,
    min_information_gain: float,
    task_type: str,
    ancestor_queries: Optional[List[str]] = None,
    ancestor_answers_df: Optional[pd.DataFrame] = None,
    max_correlation: float = 0.85,
    allow_same_attribute: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Identifies the query that maximizes Shannon Information Gain while enforcing:
    1. Attribute uniqueness: prevents re-splitting on an attribute already in this branch.
    2. Decorrelation: rejects candidates with |Pearson correlation| > max_correlation.
    3. Minimum information gain constraint.
    """
    parent_entropy = compute_shannon_entropy(labels)
    best_query: Optional[str] = None
    best_gain: float = -1.0
    best_mask: Optional[np.ndarray] = None
    best_category: Optional[str] = None

    anc_queries = ancestor_queries or []
    anc_categories = {classify_query(q, task_type=task_type) for q in anc_queries}

    for query_idx, query_text in enumerate(candidate_queries):
        category = classify_query(query_text, task_type=task_type)

        # 1. Attribute-level non-redundancy check
        if not allow_same_attribute and category in anc_categories:
            continue

        column_key = f"Q{query_idx}"
        split_mask = (answers_df[column_key].values == 1)

        # 2. Empirical correlation check against ancestor query answers
        if ancestor_answers_df is not None and not ancestor_answers_df.empty:
            cand_vec = split_mask.astype(float)
            if np.std(cand_vec) > 1e-6:
                is_redundant = False
                for anc_col in ancestor_answers_df.columns:
                    anc_vec = ancestor_answers_df[anc_col].values.astype(float)
                    if np.std(anc_vec) > 1e-6:
                        corr = abs(float(np.corrcoef(cand_vec, anc_vec)[0, 1]))
                        if not np.isnan(corr) and corr > max_correlation:
                            is_redundant = True
                            break
                if is_redundant:
                    continue

        gain, _, _ = compute_split_gain(labels, split_mask, parent_entropy)

        if gain > best_gain:
            best_gain = gain
            best_query = query_text
            best_mask = split_mask
            best_category = category

    if best_query is not None and best_gain >= min_information_gain:
        return {
            "query": best_query,
            "information_gain": float(best_gain),
            "split_mask": best_mask,
            "category": best_category
        }

    return None


def predict_single_instance(
    graph: nx.DiGraph,
    record_answers: pd.Series,
    query_to_column: Dict[str, str]
) -> Tuple[int, float]:
    current_node = 0
    while not graph.nodes[current_node]["is_leaf"]:
        split_query = graph.nodes[current_node]["query"]
        col_name = query_to_column[split_query]
        actual_answer = int(record_answers[col_name])

        next_node = None
        for _, neighbor_id, edge_attrs in graph.out_edges(current_node, data=True):
            if edge_attrs.get("branch") == actual_answer:
                next_node = neighbor_id
                break

        if next_node is None:
            return graph.nodes[current_node]["prediction"], graph.nodes[current_node]["prob"]
        current_node = next_node

    return graph.nodes[current_node]["prediction"], graph.nodes[current_node]["prob"]


class QueryDecisionTree:
    """Unified Decision Tree estimator over Natural Language Queries."""

    def __init__(
        self,
        regime: str = "local",
        n_subsets: int = 8,
        k_queries: int = 5,
        max_depth: int = 3,
        min_ig: float = 0.01,
        task_type: str = "contextual",
        majority_votes: int = 1,
        max_correlation: float = 0.85,
        allow_same_attribute: bool = False
    ):
        self.regime = regime
        self.n_subsets = n_subsets
        self.k_queries = k_queries
        self.max_depth = max_depth
        self.min_ig = min_ig
        self.task_type = task_type
        self.majority_votes = majority_votes
        self.max_correlation = max_correlation
        self.allow_same_attribute = allow_same_attribute

        self.graph: Optional[nx.DiGraph] = None
        self.global_clf: Optional[DecisionTreeClassifier] = None
        self.queries: List[str] = []
        self.splits: List[Dict[str, Any]] = []

    def fit(self, train_df: pd.DataFrame, engine: FastVLLMEngine, seed: int = 42) -> "QueryDecisionTree":
        np.random.seed(seed)
        if self.regime == "local":
            self._fit_local(train_df, engine, base_seed=seed)
        else:
            self._fit_global(train_df, engine, seed=seed)
        return self

    def _fit_local(self, train_df: pd.DataFrame, engine: FastVLLMEngine, base_seed: int):
        self.graph = nx.DiGraph()

        # Root node
        root_labels = train_df["target"].values
        n_pos = int((root_labels == 1).sum())
        n_neg = len(root_labels) - n_pos
        self.graph.add_node(
            0,
            is_leaf=True,
            depth=0,
            n_samples=len(train_df),
            n_pos=n_pos,
            n_neg=n_neg,
            prediction=int(n_pos >= n_neg),
            prob=float(n_pos / len(train_df)) if len(train_df) > 0 else 0.0
        )

        current_level = [
            {
                "node_id": 0,
                "sub_df": train_df,
                "depth": 0,
                "seed": base_seed,
                "ancestors": []
            }
        ]

        while current_level:
            next_level = []
            splittable_nodes = []

            for item in current_level:
                sub_df = item["sub_df"]
                depth = item["depth"]
                labels = sub_df["target"].values
                n_samples = len(labels)
                n_pos = int((labels == 1).sum())
                n_neg = n_samples - n_pos
                entropy = compute_shannon_entropy(labels)

                if depth >= self.max_depth or entropy == 0.0 or min(n_pos, n_neg) < 2:
                    continue
                splittable_nodes.append(item)

            if not splittable_nodes:
                break

            for item in splittable_nodes:
                cands = engine.generate_queries(
                    item["sub_df"],
                    n_subsets=self.n_subsets,
                    k_queries=self.k_queries,
                    seed=item["seed"] + 100 * item["depth"],
                    exclude=item["ancestors"],
                    ancestor_queries=item["ancestors"]
                )
                item["candidates"] = cands

            for item in splittable_nodes:
                node_id = item["node_id"]
                sub_df = item["sub_df"]
                labels = sub_df["target"].values
                candidates = item.get("candidates", [])
                ancestors = item["ancestors"]
                if not candidates:
                    continue

                sub_texts = sub_df["text"].tolist()
                ans_df = engine.answer_queries_cached(
                    sub_texts,
                    candidates,
                    cache_key_prefix=f"train_node_{node_id}",
                    majority_votes=self.majority_votes
                )

                anc_ans_df = None
                if ancestors:
                    anc_ans_df = engine.answer_queries_cached(
                        sub_texts,
                        ancestors,
                        cache_key_prefix=f"train_node_{node_id}_anc",
                        majority_votes=self.majority_votes
                    )

                split_info = find_optimal_split_query(
                    candidate_queries=candidates,
                    answers_df=ans_df,
                    labels=labels,
                    min_information_gain=self.min_ig,
                    task_type=self.task_type,
                    ancestor_queries=ancestors,
                    ancestor_answers_df=anc_ans_df,
                    max_correlation=self.max_correlation,
                    allow_same_attribute=self.allow_same_attribute
                )

                if split_info is not None:
                    chosen_query = split_info["query"]
                    split_mask = split_info["split_mask"]

                    self.graph.nodes[node_id].update(
                        is_leaf=False,
                        query=chosen_query,
                        ig=split_info["information_gain"],
                        category=split_info["category"]
                    )

                    yes_subset = sub_df[split_mask].copy()
                    no_subset = sub_df[~split_mask].copy()
                    updated_ancestors = ancestors + [chosen_query]

                    left_id = len(self.graph)
                    l_labels = yes_subset["target"].values
                    l_pos = int((l_labels == 1).sum()) if len(l_labels) > 0 else 0
                    l_neg = len(l_labels) - l_pos
                    self.graph.add_edge(node_id, left_id, branch=1)
                    self.graph.add_node(
                        left_id,
                        is_leaf=True,
                        depth=item["depth"] + 1,
                        n_samples=len(l_labels),
                        n_pos=l_pos,
                        n_neg=l_neg,
                        prediction=int(l_pos >= l_neg),
                        prob=float(l_pos / len(l_labels)) if len(l_labels) > 0 else 0.0
                    )
                    next_level.append({
                        "node_id": left_id,
                        "sub_df": yes_subset,
                        "depth": item["depth"] + 1,
                        "seed": item["seed"] + 1,
                        "ancestors": updated_ancestors
                    })

                    right_id = len(self.graph)
                    r_labels = no_subset["target"].values
                    r_pos = int((r_labels == 1).sum()) if len(r_labels) > 0 else 0
                    r_neg = len(r_labels) - r_pos
                    self.graph.add_edge(node_id, right_id, branch=0)
                    self.graph.add_node(
                        right_id,
                        is_leaf=True,
                        depth=item["depth"] + 1,
                        n_samples=len(r_labels),
                        n_pos=r_pos,
                        n_neg=r_neg,
                        prediction=int(r_pos >= r_neg),
                        prob=float(r_pos / len(r_labels)) if len(r_labels) > 0 else 0.0
                    )
                    next_level.append({
                        "node_id": right_id,
                        "sub_df": no_subset,
                        "depth": item["depth"] + 1,
                        "seed": item["seed"] + 2,
                        "ancestors": updated_ancestors
                    })

            current_level = next_level

        self.queries = list(dict.fromkeys(
            data["query"]
            for _, data in self.graph.nodes(data=True)
            if not data["is_leaf"]
        ))
        self.splits = [
            {
                "depth": data["depth"],
                "query": data["query"],
                "category": data["category"],
                "ig": data["ig"]
            }
            for _, data in self.graph.nodes(data=True)
            if not data["is_leaf"]
        ]

    def _fit_global(self, train_df: pd.DataFrame, engine: FastVLLMEngine, seed: int):
        self.queries = engine.generate_queries(train_df, self.n_subsets, self.k_queries, seed=seed)
        ans_df = engine.answer_queries_cached(
            train_df["text"].tolist(), 
            self.queries, 
            cache_key_prefix="train",
            majority_votes=self.majority_votes
        )

        self.global_clf = DecisionTreeClassifier(max_depth=self.max_depth, random_state=seed)
        self.global_clf.fit(ans_df.values, train_df["target"].values)

        importances = self.global_clf.feature_importances_
        used_indices = np.where(importances > 0.0)[0]
        self.splits = [
            {
                "depth": 0,
                "query": self.queries[idx],
                "category": classify_query(self.queries[idx], self.task_type),
                "importance": float(importances[idx])
            }
            for idx in used_indices
        ]

    def predict(self, test_df: pd.DataFrame, engine: FastVLLMEngine) -> Tuple[np.ndarray, np.ndarray]:
        if not self.queries:
            fallback_pred = self.graph.nodes[0]["prediction"] if (self.graph and len(self.graph)) else 0
            fallback_prob = self.graph.nodes[0]["prob"] if (self.graph and len(self.graph)) else 0.0
            return np.full(len(test_df), fallback_pred), np.full(len(test_df), fallback_prob)

        ans_df = engine.answer_queries_cached(
            test_df["text"].tolist(), 
            self.queries, 
            cache_key_prefix="test",
            majority_votes=self.majority_votes
        )

        if self.regime == "global":
            y_pred = self.global_clf.predict(ans_df.values)
            y_prob = self.global_clf.predict_proba(ans_df.values)[:, 1] if hasattr(self.global_clf, "predict_proba") else y_pred
            return y_pred, y_prob

        query_to_column = {q: f"Q{i}" for i, q in enumerate(self.queries)}
        predictions, probabilities = [], []

        for row_idx in range(len(ans_df)):
            pred, prob = predict_single_instance(self.graph, ans_df.iloc[row_idx], query_to_column)
            predictions.append(pred)
            probabilities.append(prob)

        return np.array(predictions), np.array(probabilities)

    def evaluate(self, test_df: pd.DataFrame, engine: FastVLLMEngine) -> Dict[str, Any]:
        y_pred, y_prob = self.predict(test_df, engine)
        metrics = compute_metrics(
            y_true=test_df["target"].values,
            y_pred=y_pred,
            y_prob=y_prob,
            splits=self.splits,
            task_type=self.task_type
        )
        metrics["splits_analysis"] = self.splits
        return metrics

    def to_mermaid(self) -> str:
        if not self.graph or len(self.graph) == 0:
            return "graph TD\nN0[Empty Tree]"

        mermaid_lines = []
        for node_id, data in self.graph.nodes(data=True):
            if data["is_leaf"]:
                mermaid_lines.append(
                    f'N{node_id}["Class: {data["prediction"]}<br/>'
                    f'(Pos={data["n_pos"]}, Neg={data["n_neg"]})"]:::leaf'
                )
            else:
                cat_tag = f'[{data["category"]}] ' if data.get("category") else ""
                clean_q = data["query"].replace('"', "'").splitlines()[0].strip()
                mermaid_lines.append(
                    f'N{node_id}["{cat_tag}{clean_q}<br/>'
                    f'IG: {data["ig"]:.3f} | N={data["n_samples"]}"]'
                )

        for u, v, edge_data in self.graph.edges(data=True):
            branch_val = edge_data.get("branch", 0)
            branch_label = "Yes" if branch_val == 1 else "No"
            mermaid_lines.append(f"N{u} -->|{branch_label}| N{v}")

        return "\n".join(mermaid_lines)



