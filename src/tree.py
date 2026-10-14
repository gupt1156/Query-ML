"""
Inductive Decision Tree Estimator over Natural Language Queries.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.generator import FastVLLMEngine
from src.utils import (
    classify_query,
    compute_metrics,
    compute_vote_consistency,
    normalize_query_for_dedup,
    parse_query_semantics,
)


def compute_shannon_entropy(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    p = float(np.mean(labels))
    return 0.0 if (p <= 0.0 or p >= 1.0) else float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))


def compute_split_gain(labels: np.ndarray, binary_mask: np.ndarray, parent_entropy: float) -> Tuple[float, int, int]:
    total = len(labels)
    pos_mask, neg_mask = (binary_mask == 1), (binary_mask == 0)
    c_pos, c_neg = int(pos_mask.sum()), int(neg_mask.sum())
    if c_pos == 0 or c_neg == 0:
        return 0.0, c_pos, c_neg
    child_ent = (c_pos / total) * compute_shannon_entropy(labels[pos_mask]) + (c_neg / total) * compute_shannon_entropy(labels[neg_mask])
    return float(parent_entropy - child_ent), c_pos, c_neg


def is_logically_consistent_with_branch(candidate_query: str, branch_history: List[Tuple[str, int]]) -> bool:
    """Verifies that a candidate query is neither a paraphrase nor logically contradictory along the branch."""
    if not branch_history:
        return True
    cand_norm = normalize_query_for_dedup(candidate_query)
    cand_sem = parse_query_semantics(candidate_query)

    for anc_query, branch_taken in branch_history:
        if normalize_query_for_dedup(anc_query) == cand_norm:
            return False
        anc_sem = parse_query_semantics(anc_query)
        if cand_sem is not None and anc_sem is not None:
            cat, op, val = cand_sem
            anc_cat, anc_op, anc_val = anc_sem
            if cat == anc_cat:
                if op == "==" or anc_op == "==":
                    return False
                if op == ">=" and anc_op == ">=":
                    if val == anc_val or (branch_taken == 0 and val >= anc_val) or (branch_taken == 1 and val <= anc_val):
                        return False
    return True


def find_optimal_split_query(
    candidate_queries: List[str],
    answers_df: pd.DataFrame,
    labels: np.ndarray,
    min_information_gain: float,
    raw_votes_dict: Optional[Dict[str, Optional[np.ndarray]]] = None,
    min_unanimous_consistency: float = 0.80,
    min_high_consistency: float = 0.90,
    min_leaf_samples: int = 2,
) -> Optional[Dict[str, Any]]:
    parent_entropy = compute_shannon_entropy(labels)
    best_split: Optional[Dict[str, Any]] = None
    best_score: float = -1.0

    for idx, q_text in enumerate(candidate_queries):
        votes_mat = raw_votes_dict.get(q_text) if raw_votes_dict else None
        consistency, unanimous_frac, high_agree_frac = compute_vote_consistency(votes_mat)
        if unanimous_frac < min_unanimous_consistency or high_agree_frac < min_high_consistency:
            continue

        split_mask = answers_df[f"Q{idx}"].values == 1
        gain, c_pos, c_neg = compute_split_gain(labels, split_mask, parent_entropy)
        if min(c_pos, c_neg) < min_leaf_samples or gain < min_information_gain:
            continue

        score = gain * (consistency ** 2)
        if score > best_score:
            best_score = score
            best_split = {
                "query": q_text,
                "information_gain": float(gain),
                "consistency": float(consistency),
                "split_mask": split_mask,
                "category": classify_query(q_text),
            }
    return best_split


class QueryDecisionTree:
    """Inductive Decision Tree estimator over Natural Language Queries."""

    def __init__(
        self,
        n_subsets: int = 8,
        k_queries: int = 5,
        max_depth: int = 3,
        min_ig: float = 0.01,
        majority_votes: int = 1,
        min_unanimous_consistency: float = 0.80,
        min_high_consistency: float = 0.90,
        min_error_reduction: int = 2,
    ):
        self.n_subsets = n_subsets
        self.k_queries = k_queries
        self.max_depth = max_depth
        self.min_ig = min_ig
        self.majority_votes = majority_votes
        self.min_unanimous_consistency = min_unanimous_consistency
        self.min_high_consistency = min_high_consistency
        self.min_error_reduction = min_error_reduction

        self.nodes: Dict[int, Dict[str, Any]] = {}
        self.queries: List[str] = []
        self.splits: List[Dict[str, Any]] = []

    def _add_leaf_node(self, node_id: int, depth: int, labels: np.ndarray) -> None:
        n_samples = len(labels)
        n_pos = int((labels == 1).sum()) if n_samples > 0 else 0
        n_neg = n_samples - n_pos
        self.nodes[node_id] = {
            "is_leaf": True,
            "depth": depth,
            "n_samples": n_samples,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "prediction": int(n_pos >= n_neg),
            "prob": float(n_pos / n_samples) if n_samples > 0 else 0.0,
        }

    def _prune_tree(self) -> None:
        changed = True
        while changed and len(self.nodes) > 1:
            changed = False
            internal = sorted(
                [(nid, d.get("depth", 0)) for nid, d in self.nodes.items() if not d.get("is_leaf", True)],
                key=lambda x: x[1],
                reverse=True,
            )
            for nid, _ in internal:
                parent = self.nodes.get(nid)
                if not parent or parent.get("is_leaf", True):
                    continue
                c0, c1 = self.nodes.get(parent.get("yes_child")), self.nodes.get(parent.get("no_child"))
                if c0 and c1 and c0.get("is_leaf", True) and c1.get("is_leaf", True):
                    err_red = min(parent["n_pos"], parent["n_neg"]) - (min(c0["n_pos"], c0["n_neg"]) + min(c1["n_pos"], c1["n_neg"]))
                    if c0["prediction"] == c1["prediction"] or err_red < self.min_error_reduction:
                        self.nodes.pop(parent["yes_child"], None)
                        self.nodes.pop(parent["no_child"], None)
                        parent["is_leaf"] = True
                        for k in ("query", "ig", "category", "consistency", "yes_child", "no_child"):
                            parent.pop(k, None)
                        changed = True

    def fit(self, train_df: pd.DataFrame, engine: FastVLLMEngine, seed: int = 42) -> "QueryDecisionTree":
        np.random.seed(seed)
        self.nodes = {}
        self._add_leaf_node(0, depth=0, labels=train_df["target"].values)
        current_level = [{"node_id": 0, "sub_df": train_df, "depth": 0, "seed": seed, "branch_history": []}]

        while current_level:
            next_level, splittable = [], []
            for item in current_level:
                labels = item["sub_df"]["target"].values
                n_pos = int((labels == 1).sum())
                if (
                    item["depth"] < self.max_depth
                    and compute_shannon_entropy(labels) > 0.0
                    and min(n_pos, len(labels) - n_pos) >= max(2, self.min_error_reduction)
                ):
                    splittable.append(item)

            if not splittable:
                break
            engine.generate_queries_for_nodes(splittable, n_subsets=self.n_subsets, k_queries=self.k_queries)

            for item in splittable:
                node_id, sub_df, branch_history = item["node_id"], item["sub_df"], item["branch_history"]
                labels = sub_df["target"].values
                candidates = [q for q in item.get("candidates", []) if is_logically_consistent_with_branch(q, branch_history)]
                if not candidates:
                    continue

                texts = sub_df["text"].tolist()
                ans_df = engine.answer_queries_cached(texts, candidates, majority_votes=self.majority_votes)
                raw_votes = {q: engine.get_votes_matrix(texts, q, self.majority_votes) for q in candidates}
                split_info = find_optimal_split_query(
                    candidates, ans_df, labels, self.min_ig, raw_votes, self.min_unanimous_consistency, self.min_high_consistency
                )

                if split_info is not None:
                    q_chosen, mask = split_info["query"], split_info["split_mask"]
                    self.nodes[node_id].update(
                        is_leaf=False, query=q_chosen, ig=split_info["information_gain"],
                        category=split_info["category"], consistency=split_info["consistency"],
                    )
                    for branch_val, edge_key, child_df, offset in (
                        (1, "yes_child", sub_df[mask].copy(), 1),
                        (0, "no_child", sub_df[~mask].copy(), 2),
                    ):
                        child_id = max(self.nodes) + 1
                        self.nodes[node_id][edge_key] = child_id
                        self._add_leaf_node(child_id, depth=item["depth"] + 1, labels=child_df["target"].values)
                        next_level.append({
                            "node_id": child_id, "sub_df": child_df, "depth": item["depth"] + 1,
                            "seed": item["seed"] + offset, "branch_history": branch_history + [(q_chosen, branch_val)],
                        })
            current_level = next_level

        self._prune_tree()
        internal = [d for _, d in sorted(self.nodes.items(), key=lambda x: (x[1]["depth"], x[0])) if not d["is_leaf"]]
        self.queries = list(dict.fromkeys(d["query"] for d in internal))
        self.splits = [{k: d[k] for k in ("depth", "query", "category", "ig", "consistency")} for d in internal]
        return self

    def predict(self, test_df: pd.DataFrame, engine: FastVLLMEngine) -> Tuple[np.ndarray, np.ndarray]:
        n = len(test_df)
        if not self.queries or not self.nodes:
            return np.full(n, self.nodes[0]["prediction"] if self.nodes else 0), np.full(n, self.nodes[0]["prob"] if self.nodes else 0.0)

        ans_mat = engine.answer_queries_cached(test_df["text"].tolist(), self.queries, majority_votes=self.majority_votes).values
        q_to_col = {q: i for i, q in enumerate(self.queries)}
        preds, probs = np.zeros(n, dtype=int), np.zeros(n, dtype=float)

        for i in range(n):
            curr = 0
            while curr in self.nodes and not self.nodes[curr]["is_leaf"]:
                ans = int(ans_mat[i, q_to_col[self.nodes[curr]["query"]]])
                nxt = self.nodes[curr].get("yes_child" if ans == 1 else "no_child")
                if nxt is None or nxt not in self.nodes:
                    break
                curr = nxt
            leaf = self.nodes.get(curr, self.nodes[0])
            preds[i], probs[i] = leaf["prediction"], leaf["prob"]
        return preds, probs

    def evaluate(self, test_df: pd.DataFrame, engine: FastVLLMEngine) -> Dict[str, Any]:
        y_pred, y_prob = self.predict(test_df, engine)
        return compute_metrics(test_df["target"].values, y_pred, y_prob, splits=self.splits)

    def to_mermaid(self) -> str:
        if not self.nodes:
            return "graph TD\nN0[Empty Tree]"
        lines = []
        for nid, d in sorted(self.nodes.items()):
            if d["is_leaf"]:
                lines.append(f'N{nid}["Class: {d["prediction"]}<br/>(Pos={d["n_pos"]}, Neg={d["n_neg"]})"]:::leaf')
            else:
                q_clean = d["query"].replace('"', "'").splitlines()[0].strip()
                lines.append(f'N{nid}["[{d["category"]}] {q_clean}<br/>IG: {d["ig"]:.3f} | N={d["n_samples"]}"]')
        for nid, d in sorted(self.nodes.items()):
            if not d["is_leaf"]:
                if "yes_child" in d:
                    lines.append(f'N{nid} -->|Yes| N{d["yes_child"]}')
                if "no_child" in d:
                    lines.append(f'N{nid} -->|No| N{d["no_child"]}')
        return "\n".join(lines)
