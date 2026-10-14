#!/usr/bin/env python3
"""Multi-Seed Inductive Decision Tree Runner for Query-ML."""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.generator import FastVLLMEngine
from src.tree import QueryDecisionTree
from src.utils import compute_side_reliability, load_admissions_data


def main():
    parser = argparse.ArgumentParser(description="Query-ML Decision Tree Runner", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seeds", type=str, default="42,52,62")
    parser.add_argument("--local_m", type=int, default=6)
    parser.add_argument("--k_queries", type=int, default=5)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--min_ig", type=float, default=0.01)
    parser.add_argument("--majority_votes", type=int, default=5)
    parser.add_argument("--min_unanimous_consistency", type=float, default=0.80)
    parser.add_argument("--min_high_consistency", type=float, default=0.90)
    parser.add_argument("--min_error_reduction", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="results/vllm_regimes")
    args = parser.parse_args()

    train_df, val_df, test_df = load_admissions_data()
    engine = FastVLLMEngine(model_name=args.model_name)

    for seed in [int(s.strip()) for s in args.seeds.split(",") if s.strip()]:
        t0 = time.time()
        tree_model = QueryDecisionTree(
            n_subsets=args.local_m,
            k_queries=args.k_queries,
            max_depth=args.max_depth,
            min_ig=args.min_ig,
            majority_votes=args.majority_votes,
            min_unanimous_consistency=args.min_unanimous_consistency,
            min_high_consistency=args.min_high_consistency,
            min_error_reduction=args.min_error_reduction,
        ).fit(train_df, engine, seed=seed)

        res = tree_model.evaluate(test_df, engine)
        res["seed"] = seed
        if val_df is not None and len(val_df) > 0:
            val_metrics = tree_model.evaluate(val_df, engine)
            res["val_accuracy"], res["val_roc_auc"] = val_metrics["accuracy"], val_metrics["roc_auc"]

        res["mermaid_diagram"] = tree_model.to_mermaid()
        res["reliability"] = compute_side_reliability(tree_model.queries, test_df["raw"].tolist(), engine, majority_votes=args.majority_votes)
        elapsed = time.time() - t0
        rel = res.get("reliability", {})
        print(
            f"[Seed {seed}] TestAcc={res['accuracy']:.4f}, ValAcc={res.get('val_accuracy', 0.0):.4f}, "
            f"AUC={res['roc_auc']:.4f}, CausalRecovery={res['causal_recovery_rate']}%, "
            f"CausalSplits={res['causal_split_pct']}% ({res['causal_splits']}/{res['total_splits']}) | "
            f"Fidelity={rel.get('fidelity_majority_pct', 0.0):.1f}%, "
            f"Consistency={rel.get('consistency_across_runs_pct', 0.0):.1f}% in {elapsed:.1f}s"
        )

        seed_dir = f"{args.output_dir}_contextual_seed{seed}"
        os.makedirs(seed_dir, exist_ok=True)
        with open(os.path.join(seed_dir, "full_experiment_data.json"), "w") as fp:
            json.dump({"config": vars(args), "timing": {"local_seconds": elapsed}, "local_runs": [res]}, fp, indent=2)


if __name__ == "__main__":
    main()
