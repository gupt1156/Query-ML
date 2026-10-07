#!/usr/bin/env python3
"""
Multi-Seed Experiment Runner for Query-ML.

This script executes empirical comparisons between:
1. Local Regime: Inductive decision trees with on-demand local query generation.
2. Global Regime: Decision tree classifier on a static global query bank.

Evaluates test accuracy, ROC-AUC, causal recovery rates, and distractor splits
across multiple seeds and benchmarks (Single-Track and Dual-Track Admissions).
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import load_admissions_data
from src.generator import FastVLLMEngine
from src.reliability import compute_side_reliability
from src.tree import QueryDecisionTree


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query-ML: Statistical LLM Query-Based Decision Tree Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    data_group = parser.add_argument_group("Benchmark Dataset Options")
    data_group.add_argument(
        "--task_type",
        type=str,
        default="contextual",
        choices=["contextual", "original"],
        help="Target benchmark: 'contextual' (Dual-Track) or 'original' (Single-Track)"
    )
    data_group.add_argument(
        "--seeds",
        type=str,
        default="42,52,62",
        help="Comma-separated list of random seeds to evaluate (e.g. '42,52,62,72')"
    )

    tree_group = parser.add_argument_group("Decision Tree Hyperparameters")
    tree_group.add_argument(
        "--global_M",
        type=int,
        default=8,
        help="Number of contrastive subsets sampled globally upfront (M)"
    )
    tree_group.add_argument(
        "--local_m",
        type=int,
        default=8,
        help="Number of contrastive subsets sampled per local tree node (m)"
    )
    tree_group.add_argument(
        "--k_queries",
        type=int,
        default=5,
        help="Number of candidate queries generated per contrastive subset (k)"
    )
    tree_group.add_argument(
        "--max_depth",
        type=int,
        default=3,
        help="Maximum tree depth"
    )
    tree_group.add_argument(
        "--min_ig",
        type=float,
        default=0.02,
        help="Minimum Shannon Information Gain threshold required to split a node"
    )
    tree_group.add_argument(
        "--majority_votes",
        type=int,
        default=1,
        help="Number of stochastic passes for majority voting (1 = greedy T=0.0)"
    )
    tree_group.add_argument(
        "--max_correlation",
        type=float,
        default=0.85,
        help="Maximum Pearson correlation permitted between candidate query and ancestor queries"
    )
    tree_group.add_argument(
        "--allow_same_attribute",
        action="store_true",
        default=False,
        help="Allow re-splitting on the same attribute along the same tree branch"
    )

    model_group = parser.add_argument_group("Inference & Output Options")
    model_group.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model identifier or local snapshot path"
    )
    model_group.add_argument(
        "--output_dir",
        type=str,
        default="results/vllm_regimes",
        help="Base directory path for persisting experiment JSON results"
    )
    model_group.add_argument(
        "--regimes",
        type=str,
        default="local",
        help="Comma-separated list of regimes to evaluate ('local', 'global', or 'global,local')"
    )

    return parser


def evaluate_single_regime(
    regime: str,
    num_subsets: int,
    seed: int,
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    engine: FastVLLMEngine
) -> Tuple[Dict[str, Any], float]:
    start_time = time.time()

    tree_model = QueryDecisionTree(
        regime=regime,
        n_subsets=num_subsets,
        k_queries=args.k_queries,
        max_depth=args.max_depth,
        min_ig=args.min_ig,
        task_type=args.task_type,
        majority_votes=args.majority_votes,
        max_correlation=args.max_correlation,
        allow_same_attribute=args.allow_same_attribute
    )

    tree_model.fit(train_df, engine, seed=seed)
    evaluation_results = tree_model.evaluate(test_df, engine)
    evaluation_results["seed"] = seed

    if regime == "local":
        evaluation_results["mermaid_diagram"] = tree_model.to_mermaid()

    # Collect Ground-Truth Fidelity and Consistency Across Runs on the side
    rel_metrics = compute_side_reliability(
        queries=tree_model.queries,
        raw_records=test_df["raw"].tolist(),
        engine=engine,
        cache_prefix="test",
        majority_votes=args.majority_votes
    )
    evaluation_results["reliability"] = rel_metrics

    elapsed_time = time.time() - start_time
    return evaluation_results, elapsed_time


def persist_seed_results(
    output_dir: str,
    task_type: str,
    seed: int,
    config: Dict[str, Any],
    timings: Dict[str, float],
    results: Dict[str, Any]
) -> str:
    seed_output_dir = f"{output_dir}_{task_type}_seed{seed}"
    os.makedirs(seed_output_dir, exist_ok=True)

    payload = {
        "config": config,
        "timing": timings,
        "global_runs": [results["global"]] if "global" in results else [],
        "local_runs": [results["local"]] if "local" in results else []
    }

    file_path = os.path.join(seed_output_dir, "full_experiment_data.json")
    with open(file_path, "w") as fp:
        json.dump(payload, fp, indent=2)

    return file_path


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    seed_list: List[int] = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print("=" * 80)
    print(f"QUERY-ML EXPERIMENT RUNNER")
    print(f"Task: {args.task_type.upper()} | Evaluated Seeds: {seed_list}")
    print(f"Global M={args.global_M}, Local m={args.local_m}, k={args.k_queries}, max_depth={args.max_depth}, min_ig={args.min_ig}")
    print(f"Non-Redundancy: max_corr={args.max_correlation}, allow_same_attr={args.allow_same_attribute}, majority_votes={args.majority_votes}")
    print("=" * 80)

    train_df, test_df, metadata = load_admissions_data(task_type=args.task_type)
    print(f"Loaded {len(train_df)} training records, {len(test_df)} test records.")

    engine = FastVLLMEngine(model_name=args.model_name)

    for seed in seed_list:
        print(f"\n" + "-" * 60)
        print(f">>> RUNNING SEED {seed} ({args.task_type.upper()}) <<<")
        print("-" * 60)

        regime_results: Dict[str, Any] = {}
        regime_timings: Dict[str, float] = {}

        regimes_to_run = [r.strip().lower() for r in args.regimes.split(",") if r.strip()]
        regime_configs = []
        if "global" in regimes_to_run:
            regime_configs.append(("global", args.global_M))
        if "local" in regimes_to_run:
            regime_configs.append(("local", args.local_m))

        for regime_name, subset_count in regime_configs:
            results, elapsed = evaluate_single_regime(
                regime=regime_name,
                num_subsets=subset_count,
                seed=seed,
                args=args,
                train_df=train_df,
                test_df=test_df,
                engine=engine
            )

            regime_results[regime_name] = results
            regime_timings[f"{regime_name}_seconds"] = elapsed

            rel = results.get("reliability", {})
            print(
                f"[{regime_name.upper()} Seed {seed}] "
                f"Acc={results['accuracy']:.4f}, "
                f"AUC={results['roc_auc']:.4f}, "
                f"CausalRecovery={results['causal_recovery_rate']}% | "
                f"Reliability (Test): "
                f"Fidelity={rel.get('fidelity_majority_pct', 0.0):.1f}% (Maj) / {rel.get('fidelity_single_draw_pct', 0.0):.1f}% (Single), "
                f"Consistency={rel.get('consistency_across_runs_pct', 0.0):.1f}%, "
                f"FlipRate={rel.get('flip_rate_pct', 0.0):.1f}% "
                f"in {elapsed:.1f}s"
            )

        saved_path = persist_seed_results(
            output_dir=args.output_dir,
            task_type=args.task_type,
            seed=seed,
            config=vars(args),
            timings=regime_timings,
            results=regime_results
        )
        print(f"Results saved to: {saved_path}")

    print("\n" + "=" * 80)
    print(f"BATCH EVALUATION COMPLETED ACROSS SEEDS: {seed_list}")
    print("=" * 80)


if __name__ == "__main__":
    main()


