#!/usr/bin/env python3
"""
Sub-7B ('Small / Stupid' LLM) Accuracy, Reliability & Speedup Benchmark Suite.

Evaluates:
1. Direct LLM Prediction & Speed (WITHOUT Tree)
2. Query Simplicity Audit (Atomic vs. Legacy Complex/Ordinal Queries)
3. Tree-Guided Inference & Speed (WITH Tree across 50 learned trees)
4. Self-Induced Tree Construction & Build Speed
"""

import argparse
import glob
import json
import os
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.generator import (
    FastVLLMEngine,
    build_direct_classification_prompt,
    build_multi_answer_prompt,
    sample_contrastive_record_pairs,
)
from src.tree import QueryDecisionTree
from src.utils import (
    accuracy_score,
    compute_side_reliability,
    compute_vote_consistency,
    extract_answers_from_llm_response,
    load_admissions_data,
    roc_auc_score,
    traverse_mermaid_tree,
)

CANONICAL_SIMPLE_QUERIES = [
    "Does the record show a Math grade of A?",
    "Does the record show at least 1 publication?",
    "Does the record show at least 2 extracurricular activities?",
    "Does the record show a Math grade of B?",
]

LEGACY_COMPLEX_OR_AMBIGUOUS_QUERIES = [
    "Does the record list a Math grade of D or lower?",
    "Does the record have a Math grade of B or higher?",
    "Does the record have a Math grade of A and at least 1 publication?",
    "Does the record mention 0 extracurricular activities?",
    "Does the record list at least 2 number of extracurricular activities?",
]

STOP_TOKENS = ["\n\n", "Note:", "Explanation:", "Reasoning:", "Question:", "User:", "<|im_end|>"]


def evaluate_direct_llm_predictions_and_speed(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    engine: FastVLLMEngine,
    seeds: List[int],
    majority_votes: int = 5,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    y_true = test_df["target"].values
    test_texts = test_df["text"].tolist()
    n_test, pack_size = len(test_texts), 5

    pairs_s0 = sample_contrastive_record_pairs(train_df, num_subsets=1, samples_per_group=3, base_seed=seeds[0])
    pos_s0, neg_s0 = "\n\n".join(pairs_s0[0][0]), "\n\n".join(pairs_s0[0][1])

    t0 = time.perf_counter()
    _ = engine.generate_batch_text([build_direct_classification_prompt(pos_s0, neg_s0, [t]) for t in test_texts], max_tokens=24, temperature=temperature, n=majority_votes, stop=STOP_TOKENS)
    baseline_direct_5vote_sec = time.perf_counter() - t0

    packed_prompts = [build_direct_classification_prompt(pos_s0, neg_s0, test_texts[i : i + pack_size]) for i in range(0, n_test, pack_size)]
    fast_stop = STOP_TOKENS + [f"\n{pack_size + 1}.", f"{pack_size + 1}.", "Record", "RECORD"]

    t0 = time.perf_counter()
    _ = engine.generate_batch_text(packed_prompts, max_tokens=pack_size * 5 + 2, temperature=temperature, n=majority_votes, stop=fast_stop)
    fast_direct_5vote_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    raw_packed1 = engine.generate_batch_text(packed_prompts, max_tokens=pack_size * 5 + 2, temperature=0.0, n=1, stop=fast_stop)
    fast_direct_1vote_sec = time.perf_counter() - t0

    packed1_preds = [ans for i, txt in enumerate(raw_packed1) for ans in extract_answers_from_llm_response(txt, min(pack_size, n_test - i * pack_size))]
    fast_direct_1vote_acc = float(accuracy_score(y_true, np.array(packed1_preds[:n_test])))

    all_prompts = []
    for seed in seeds:
        pos_list, neg_list = sample_contrastive_record_pairs(train_df, num_subsets=1, samples_per_group=3, base_seed=seed)[0]
        p_blk, n_blk = "\n\n".join(pos_list), "\n\n".join(neg_list)
        all_prompts.extend(build_direct_classification_prompt(p_blk, n_blk, [t]) for t in test_texts)

    raw_multi_all = engine.generate_batch_text(all_prompts, max_tokens=6, temperature=temperature, n=majority_votes, stop=STOP_TOKENS + ["\n2.", "2."])

    seed_results = []
    for s_idx, seed in enumerate(seeds):
        raw_multi = raw_multi_all[s_idx * n_test : (s_idx + 1) * n_test]
        vote_matrix = np.array([[extract_answers_from_llm_response(c, 1)[0] for c in comps] for comps in raw_multi], dtype=int)
        pos_votes = vote_matrix.sum(axis=1)
        consist, unan, high_ag = compute_vote_consistency(vote_matrix, pairwise=True)
        seed_results.append({
            "seed": seed,
            "acc_maj5": float(accuracy_score(y_true, (pos_votes > (majority_votes / 2.0)).astype(int))),
            "acc_maj3": float(accuracy_score(y_true, (vote_matrix[:, :3].sum(axis=1) >= 2).astype(int))),
            "acc_single": float(np.mean([accuracy_score(y_true, vote_matrix[:, v]) for v in range(majority_votes)])),
            "auc_maj5": float(roc_auc_score(y_true, pos_votes / float(majority_votes))),
            "consistency": consist,
            "unanimous_rate": unan,
            "high_agree_rate": high_ag,
        })

    mean_pct = lambda key: round(float(np.mean([r[key] for r in seed_results]) * 100), 2)
    return {
        "n_seeds": len(seeds),
        "mean_acc_maj5_pct": mean_pct("acc_maj5"),
        "std_acc_maj5_pct": round(float(np.std([r["acc_maj5"] for r in seed_results]) * 100), 2),
        "mean_acc_maj3_pct": mean_pct("acc_maj3"),
        "mean_acc_single_pct": mean_pct("acc_single"),
        "fast_packed_1vote_acc_pct": round(fast_direct_1vote_acc * 100, 2),
        "mean_auc_maj5": round(float(np.mean([r["auc_maj5"] for r in seed_results])), 4),
        "mean_consistency_pct": mean_pct("consistency"),
        "mean_unanimous_pct": mean_pct("unanimous_rate"),
        "mean_high_agree_pct": mean_pct("high_agree_rate"),
        "speed_200_records": {
            "baseline_direct_5vote_sec": round(baseline_direct_5vote_sec, 3),
            "fast_packed_direct_5vote_sec": round(fast_direct_5vote_sec, 3),
            "fast_packed_direct_1vote_sec": round(fast_direct_1vote_sec, 3),
            "speedup_packed_5vote_x": round(baseline_direct_5vote_sec / max(1e-4, fast_direct_5vote_sec), 2),
            "speedup_packed_1vote_x": round(baseline_direct_5vote_sec / max(1e-4, fast_direct_1vote_sec), 2),
        },
        "per_seed": seed_results,
    }


def evaluate_legacy_vs_simple_queries(test_df: pd.DataFrame, engine: FastVLLMEngine, majority_votes: int = 5) -> Dict[str, Any]:
    texts, raw_records = test_df["text"].tolist(), test_df["raw"].tolist()
    engine.answer_queries_cached(texts, CANONICAL_SIMPLE_QUERIES + LEGACY_COMPLEX_OR_AMBIGUOUS_QUERIES, majority_votes=majority_votes)
    return {
        "simple_self_contained": compute_side_reliability(CANONICAL_SIMPLE_QUERIES, raw_records, engine, majority_votes=majority_votes),
        "legacy_complex_or_ordinal": compute_side_reliability(LEGACY_COMPLEX_OR_AMBIGUOUS_QUERIES, raw_records, engine, majority_votes=majority_votes, pairwise=True),
    }


def evaluate_tree_guided_inference_and_speed(tree_json_pattern: str, test_df: pd.DataFrame, engine: FastVLLMEngine) -> Dict[str, Any]:
    files = sorted(glob.glob(tree_json_pattern))
    if not files:
        return {}

    trees_data = [json.load(open(f))["local_runs"][0] for f in files]
    all_queries = sorted({s["query"] for r in trees_data for s in r.get("splits_analysis", [])})
    texts, y_true = test_df["text"].tolist(), test_df["target"].values
    n_test = len(texts)
    core_3_queries = CANONICAL_SIMPLE_QUERIES[:3][::-1]

    t0 = time.perf_counter()
    for q_single in core_3_queries:
        _ = engine.generate_batch_text([build_multi_answer_prompt(t, [q_single]) for t in texts], max_tokens=24, temperature=0.2, n=5, stop=STOP_TOKENS)
    baseline_tree_5vote_sec = time.perf_counter() - t0

    packed_prompts = [build_multi_answer_prompt(t, core_3_queries) for t in texts]
    tight_stop = STOP_TOKENS + ["\n4.", "4.", "Record", "RECORD", "EXAMPLE"]
    packed_times = {}
    for votes, temp in ((5, 0.2), (3, 0.2), (1, 0.0)):
        t0 = time.perf_counter()
        _ = engine.generate_batch_text(packed_prompts, max_tokens=17, temperature=temp, n=votes, stop=tight_stop)
        packed_times[votes] = time.perf_counter() - t0

    single_stop = STOP_TOKENS + ["\n2.", "2.", "Record", "RECORD", "EXAMPLE"]
    t0 = time.perf_counter()
    out0 = engine.generate_batch_text([build_multi_answer_prompt(t, [core_3_queries[0]]) for t in texts], max_tokens=6, temperature=0.0, n=1, stop=single_stop)
    ans0 = [extract_answers_from_llm_response(o, 1)[0] for o in out0]
    out1 = engine.generate_batch_text([build_multi_answer_prompt(texts[i], [core_3_queries[1] if ans0[i] == 1 else core_3_queries[2]]) for i in range(n_test)], max_tokens=6, temperature=0.0, n=1, stop=single_stop)
    ans1 = [extract_answers_from_llm_response(o, 1)[0] for o in out1]
    lvl2 = [i for i in range(n_test) if ans0[i] != ans1[i]]
    if lvl2:
        _ = engine.generate_batch_text([build_multi_answer_prompt(texts[i], [core_3_queries[2] if ans0[i] == 1 else core_3_queries[1]]) for i in lvl2], max_tokens=6, temperature=0.0, n=1, stop=single_stop)
    fast_tree_routed_1vote_sec = time.perf_counter() - t0

    engine.answer_queries_cached(texts, all_queries, majority_votes=5)
    ans_votes5 = {q: engine.get_votes_matrix(texts, q, majority_votes=5) for q in all_queries}
    ans_maj5 = {q: (v.sum(axis=1) >= 3).astype(int) for q, v in ans_votes5.items()}
    ans_maj3 = {q: (v[:, :3].sum(axis=1) >= 2).astype(int) for q, v in ans_votes5.items()}
    ans_maj1 = {q: v[:, 0].astype(int) for q, v in ans_votes5.items()}

    accs_maj5, accs_maj3, accs_maj1, tree_consist = [], [], [], []
    for r in trees_data:
        m_str = r.get("mermaid_diagram", "")
        accs_maj5.append(float(accuracy_score(y_true, traverse_mermaid_tree(m_str, ans_maj5, n_test))))
        accs_maj3.append(float(accuracy_score(y_true, traverse_mermaid_tree(m_str, ans_maj3, n_test))))
        accs_maj1.append(float(accuracy_score(y_true, traverse_mermaid_tree(m_str, ans_maj1, n_test))))
        v_preds = np.column_stack([traverse_mermaid_tree(m_str, {q: ans_votes5[q][:, v_i] for q in all_queries}, n_test) for v_i in range(5)])
        tree_consist.append(compute_vote_consistency(v_preds, pairwise=True)[0])

    return {
        "n_trees": len(trees_data),
        "tree_guided_acc_maj5_mean_pct": round(float(np.mean(accs_maj5) * 100), 2),
        "tree_guided_acc_maj5_median_pct": round(float(np.median(accs_maj5) * 100), 2),
        "tree_guided_acc_maj5_max_pct": round(float(np.max(accs_maj5) * 100), 2),
        "tree_guided_acc_maj3_mean_pct": round(float(np.mean(accs_maj3) * 100), 2),
        "tree_guided_acc_maj3_median_pct": round(float(np.median(accs_maj3) * 100), 2),
        "tree_guided_acc_maj1_mean_pct": round(float(np.mean(accs_maj1) * 100), 2),
        "tree_guided_acc_maj1_median_pct": round(float(np.median(accs_maj1) * 100), 2),
        "tree_pred_consistency_pct": round(float(np.mean(tree_consist) * 100), 2),
        "speed_200_records": {
            "baseline_tree_seq_5vote_sec": round(baseline_tree_5vote_sec, 3),
            "fast_tree_packed_5vote_sec": round(packed_times[5], 3),
            "fast_tree_packed_3vote_sec": round(packed_times[3], 3),
            "fast_tree_packed_1vote_sec": round(packed_times[1], 3),
            "fast_tree_routed_1vote_sec": round(fast_tree_routed_1vote_sec, 3),
            "avg_queries_evaluated_routed": round((2 * n_test + len(lvl2)) / float(n_test), 2),
            "speedup_packed_5vote_x": round(baseline_tree_5vote_sec / max(1e-4, packed_times[5]), 2),
            "speedup_packed_3vote_x": round(baseline_tree_5vote_sec / max(1e-4, packed_times[3]), 2),
            "speedup_packed_1vote_x": round(baseline_tree_5vote_sec / max(1e-4, packed_times[1]), 2),
            "speedup_routed_1vote_x": round(baseline_tree_5vote_sec / max(1e-4, fast_tree_routed_1vote_sec), 2),
        },
    }


def evaluate_self_induced_trees(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, engine: FastVLLMEngine, seeds: List[int], majority_votes: int = 5
) -> Dict[str, Any]:
    accs, val_accs, aucs, recovs, split_pcts, times = [], [], [], [], [], []
    for seed in seeds:
        t0 = time.perf_counter()
        tree = QueryDecisionTree(n_subsets=6, k_queries=5, majority_votes=majority_votes).fit(train_df, engine, seed=seed)
        res, val_res = tree.evaluate(test_df, engine), tree.evaluate(val_df, engine)
        times.append(time.perf_counter() - t0)
        accs.append(res["accuracy"])
        val_accs.append(val_res["accuracy"])
        aucs.append(res["roc_auc"])
        recovs.append(res.get("causal_recovery_rate", 0.0))
        split_pcts.append(res.get("causal_split_pct", 0.0))

    return {
        "n_seeds": len(seeds),
        "self_tree_test_acc_mean_pct": round(float(np.mean(accs) * 100), 2),
        "self_tree_test_acc_median_pct": round(float(np.median(accs) * 100), 2),
        "self_tree_test_acc_max_pct": round(float(np.max(accs) * 100), 2),
        "self_tree_val_acc_mean_pct": round(float(np.mean(val_accs) * 100), 2),
        "self_tree_auc_mean": round(float(np.mean(aucs)), 4),
        "self_tree_causal_recovery_pct": round(float(np.mean(recovs)), 2),
        "self_tree_causal_split_pct": round(float(np.mean(split_pcts)), 2),
        "mean_build_sec_per_seed": round(float(np.mean(times)), 2),
        "median_build_sec_per_seed": round(float(np.median(times)), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="5-Model Sub-7B LLM Accuracy, Reliability & Speedup Suite")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--tree_json_pattern", type=str, default="results/vllm_regimes_maj5_final_contextual_seed*/full_experiment_data.json")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--direct_seeds", type=str, default="42,52,62,72,82,92,102,112,122,132")
    parser.add_argument("--induce_seeds", type=str, default="42,52,62,72,82,92,102,112,122,132")
    args = parser.parse_args()

    t0 = time.time()
    train_df, val_df, test_df = load_admissions_data()
    engine = FastVLLMEngine(model_name=args.model_name)
    d_seeds = [int(s) for s in args.direct_seeds.split(",") if s.strip()]
    i_seeds = [int(s) for s in args.induce_seeds.split(",") if s.strip()]

    payload = {
        "model_name": args.model_name,
        "direct_llm_no_tree": evaluate_direct_llm_predictions_and_speed(train_df, test_df, engine, seeds=d_seeds),
        "query_simplicity_audit": evaluate_legacy_vs_simple_queries(test_df, engine),
        "tree_guided_50trees": evaluate_tree_guided_inference_and_speed(args.tree_json_pattern, test_df, engine),
        "self_induced_trees": evaluate_self_induced_trees(train_df, val_df, test_df, engine, seeds=i_seeds),
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as fp:
        json.dump(payload, fp, indent=2)


if __name__ == "__main__":
    main()
