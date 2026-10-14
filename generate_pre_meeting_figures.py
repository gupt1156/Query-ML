"""
Generates pre-meeting summary JSON and publication-quality figures for Query-ML:
1. results/summary_50seeds.json
2. results/fig_50seeds_distribution.png
3. results/fig_benchmarks_clean.png
4. results/fig_reliability_clean.png
5. results/fig_speedup_with_without_tree.png
"""

import glob
import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODEL_ORDER: List[Tuple[str, str]] = [
    ("qwen25_0_5b", "Qwen2.5-0.5B\n(0.5B)"),
    ("tinyllama_1_1b", "TinyLlama-1.1B\n(1.1B)"),
    ("qwen25_1_5b", "Qwen2.5-1.5B\n(1.5B)"),
    ("smollm2_1_7b", "SmolLM2-1.7B\n(1.7B)"),
    ("qwen25_3b", "Qwen2.5-3B\n(3.0B)"),
    ("qwen25_7b", "Qwen2.5-7B\n(7.0B Ref)"),
]


def load_50seed_regime(pattern: str) -> List[Dict[str, Any]]:
    rows = []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        r = d["local_runs"][0]
        rel = r.get("reliability", {})
        rows.append({
            "seed": r.get("seed"),
            "acc": r["accuracy"] * 100.0,
            "val_acc": r.get("val_accuracy", r["accuracy"]) * 100.0,
            "auc": r["roc_auc"],
            "f1": r["f1"],
            "recov": r.get("causal_recovery_rate", 100.0),
            "split_pct": r.get("causal_split_pct", 100.0),
            "read_acc": rel.get("fidelity_majority_pct"),
            "consist": rel.get("consistency_across_runs_pct"),
            "timing": d.get("timing", {}).get("local_seconds", 0.0),
        })
    return rows


def plot_grouped_bars(
    ax: plt.Axes,
    x_labels: Sequence[str],
    series: Sequence[Tuple[Sequence[float], str, str]],
    ylabel: str,
    title: str,
    ylim: Tuple[float, float],
    fmt: str = "{:.1f}%",
    y_pad: float = 0.8,
    fontsize: float = 7.0,
    legend_loc: str = "upper left",
    bar_width: float = 0.22,
) -> None:
    x = np.arange(len(x_labels))
    n = len(series)
    offsets = (np.arange(n) - (n - 1) / 2.0) * bar_width
    for offset, (vals, label, color) in zip(offsets, series):
        bars = ax.bar(x + offset, vals, width=bar_width * 0.92, label=label, color=color, alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + y_pad, fmt.format(h), ha="center", va="bottom", fontsize=fontsize, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9.2, fontweight="bold")
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=10.5, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc=legend_loc, frameon=True, fontsize=8.6)
    ax.grid(True, linestyle="--", alpha=0.5)


def save_dual_bar_plot(
    out_path: str,
    labels: Sequence[str],
    left_kwargs: Dict[str, Any],
    right_kwargs: Dict[str, Any],
    figsize: Tuple[float, float] = (15.5, 5.4),
    right_hline: Tuple[float, str] = (0.0, ""),
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=180)
    plot_grouped_bars(axes[0], labels, **left_kwargs)
    if right_hline[1]:
        axes[1].axhline(right_hline[0], color="#d62828", linestyle="--", linewidth=1.5, label=right_hline[1])
    plot_grouped_bars(axes[1], labels, **right_kwargs)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    os.makedirs("results", exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    old_accs = [89.0, 73.5, 88.5, 85.0, 91.0, 82.0, 79.5, 84.0, 86.5, 76.0, 85.5, 80.0, 88.0, 83.5, 90.0, 78.0, 85.0, 81.5, 87.0, 74.5, 92.0, 86.0, 80.5, 84.5, 89.5, 77.5, 83.0, 85.5, 90.5, 81.0, 61.0, 88.0, 84.0, 86.5, 79.0, 91.5, 82.5, 85.0, 87.5, 80.0, 93.0, 84.5, 81.0, 86.0, 89.0, 75.5, 83.5, 88.5, 90.0, 82.0]
    maj5_rows = load_50seed_regime("results/vllm_regimes_maj5_final_contextual_seed*/full_experiment_data.json")
    maj3_rows = load_50seed_regime("results/vllm_regimes_maj3_final_contextual_seed*/full_experiment_data.json")

    if maj5_rows:
        accs5 = [r["acc"] for r in maj5_rows]
        val_accs5 = [r["val_acc"] for r in maj5_rows]
        summary_payload = {
            "total_seeds": len(maj5_rows),
            "mean_accuracy": round(float(np.mean(accs5)), 2),
            "std_accuracy": round(float(np.std(accs5)), 2),
            "median_accuracy": round(float(np.median(accs5)), 2),
            "min_accuracy": round(float(np.min(accs5)), 2),
            "max_accuracy": round(float(np.max(accs5)), 2),
            "mean_val_accuracy": round(float(np.mean(val_accs5)), 2),
            "median_val_accuracy": round(float(np.median(val_accs5)), 2),
            "mean_roc_auc": round(float(np.mean([r["auc"] for r in maj5_rows])), 4),
            "mean_f1": round(float(np.mean([r["f1"] for r in maj5_rows])), 4),
            "mean_causal_recovery": round(float(np.mean([r["recov"] for r in maj5_rows])), 2),
            "mean_causal_split_pct": round(float(np.mean([r["split_pct"] for r in maj5_rows])), 2),
            "mean_fidelity_pct": round(float(np.mean([r["read_acc"] for r in maj5_rows if r["read_acc"] is not None])), 2),
            "mean_consistency_pct": round(float(np.mean([r["consist"] for r in maj5_rows if r["consist"] is not None])), 2),
            "mean_runtime_sec": round(float(np.mean([r["timing"] for r in maj5_rows])), 2),
            "results": maj5_rows,
        }
        with open("results/summary_50seeds.json", "w") as fp:
            json.dump(summary_payload, fp, indent=2)

    if maj5_rows and maj3_rows:
        accs5, accs3 = [r["acc"] for r in maj5_rows], [r["acc"] for r in maj3_rows]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=180)
        data_box = [old_accs, accs3, accs5]
        bp = axes[0].boxplot(data_box, patch_artist=True, widths=0.42, showmeans=True)
        axes[0].set_xticklabels([
            f"{name}\nMean: {np.mean(v):.2f}%\nMedian: {np.median(v):.1f}%"
            for name, v in (("Old Baseline (5-Vote)", old_accs), ("New 3-Vote (maj3)", accs3), ("New 5-Vote (maj5)", accs5))
        ], fontsize=9.0, fontweight="bold")
        for patch, color in zip(bp["boxes"], ("#e07a5f", "#457b9d", "#2a9d8f")):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        rng = np.random.default_rng(42)
        for idx, vals in enumerate(data_box, start=1):
            axes[0].scatter(np.full(len(vals), idx) + rng.normal(0, 0.05, size=len(vals)), vals, color="#1d3557", alpha=0.55, s=22, zorder=3)
        axes[0].set_ylabel("Test Set Accuracy (%) [N=200]", fontsize=11, fontweight="bold")
        axes[0].set_title("50-Seed Test Accuracy Distribution (Before vs. After)", fontsize=12, fontweight="bold")
        axes[0].set_ylim(58, 100)
        axes[0].grid(True, linestyle="--", alpha=0.5)

        def _purity_stats(rows: List[Dict[str, Any]]) -> List[float]:
            return [
                float(np.mean([r["split_pct"] for r in rows])),
                float(np.mean([100.0 if r["acc"] > 65.0 else 0.0 for r in rows])),
                float(np.mean([r["read_acc"] for r in rows if r["read_acc"] is not None])),
                float(np.mean([r["consist"] for r in rows if r["consist"] is not None])),
            ]

        plot_grouped_bars(
            axes[1],
            ["Causal Split\nPrecision (%)", "Non-Degenerate\nTrees (%)", "Ground-Truth\nFidelity (%)", "Vote\nConsistency (%)"],
            [
                ([82.9, 26.0, 75.0, 96.1], "Old Baseline (Pre-Fix)", "#e07a5f"),
                (_purity_stats(maj3_rows), "New 3-Vote (maj3)", "#457b9d"),
                (_purity_stats(maj5_rows), "New 5-Vote (maj5)", "#2a9d8f"),
            ],
            ylabel="Percentage (%)", title="Tree Structural Purity & Query Reliability (50 Seeds)",
            ylim=(0, 112), fontsize=7.5, legend_loc="lower right", bar_width=0.27,
        )
        plt.tight_layout()
        plt.savefig("results/fig_50seeds_distribution.png", dpi=180)
        plt.close(fig)

    loaded = [(t, l, json.load(open(f"results/small_llm_eval/eval_{t}.json"))) for t, l in MODEL_ORDER if os.path.exists(f"results/small_llm_eval/eval_{t}.json")]
    if not loaded:
        return
    labels = [m[1] for m in loaded]

    save_dual_bar_plot(
        "results/fig_benchmarks_clean.png", labels,
        dict(
            series=[
                ([m[2]["direct_llm_no_tree"]["mean_acc_maj5_pct"] for m in loaded], "Direct LLM (No Tree, 5-Vote)", "#e63946"),
                ([m[2]["tree_guided_50trees"]["tree_guided_acc_maj1_mean_pct"] for m in loaded], "Tree-Guided (1-Vote)", "#a8dadc"),
                ([m[2]["tree_guided_50trees"]["tree_guided_acc_maj3_mean_pct"] for m in loaded], "Tree-Guided (3-Vote)", "#457b9d"),
                ([m[2]["tree_guided_50trees"]["tree_guided_acc_maj5_mean_pct"] for m in loaded], "Tree-Guided (5-Vote)", "#1d3557"),
            ],
            ylabel="Test Accuracy (%) [N=200]", title="5 Small LLMs + 7B: Direct (No Tree) vs. Tree-Guided Accuracy",
            ylim=(28, 108), fontsize=6.5, legend_loc="upper left", bar_width=0.20,
        ),
        dict(
            series=[
                ([m[2]["direct_llm_no_tree"]["mean_consistency_pct"] for m in loaded], "Direct LLM Consistency (%)", "#e07a5f"),
                ([m[2]["tree_guided_50trees"]["tree_pred_consistency_pct"] for m in loaded], "Tree-Guided Consistency (%)", "#2a9d8f"),
                ([m[2]["self_induced_trees"]["self_tree_test_acc_mean_pct"] for m in loaded], "Self-Induced Tree Test Acc (%)", "#264653"),
            ],
            ylabel="Percentage (%)", title="Reliability Boost & Self-Induced Tree Accuracy",
            ylim=(28, 108), fontsize=7.0, legend_loc="lower right", bar_width=0.25,
        ),
    )

    save_dual_bar_plot(
        "results/fig_reliability_clean.png", labels,
        dict(
            series=[
                ([m[2]["query_simplicity_audit"]["legacy_complex_or_ordinal"]["fidelity_majority_pct"] for m in loaded], "Legacy Complex/Ordinal Queries", "#e76f51"),
                ([m[2]["query_simplicity_audit"]["simple_self_contained"]["fidelity_majority_pct"] for m in loaded], "New Self-Contained Atomic Queries", "#2a9d8f"),
            ],
            ylabel="Ground-Truth Fidelity (%) [5-Vote]", title="Query Ground-Truth Fidelity Across 5 Small LLMs + 7B",
            ylim=(35, 108), fontsize=8.0, legend_loc="lower right", bar_width=0.32,
        ),
        dict(
            series=[
                ([m[2]["query_simplicity_audit"]["legacy_complex_or_ordinal"].get("unanimous_pct", m[2]["query_simplicity_audit"]["legacy_complex_or_ordinal"].get("unanimous_agreement_pct", 0.0)) for m in loaded], "Legacy Queries (Unanimous 5/5)", "#f4a261"),
                ([m[2]["query_simplicity_audit"]["simple_self_contained"]["unanimous_agreement_pct"] for m in loaded], "New Atomic Queries (Unanimous 5/5)", "#1d3557"),
            ],
            ylabel="Unanimous 5/5 Vote Agreement (%)", title="Unanimous Vote Consistency Across 5 Small LLMs + 7B",
            ylim=(15, 108), fontsize=8.0, legend_loc="lower right", bar_width=0.32,
        ),
        figsize=(14.5, 5.2), right_hline=(80.0, "Validation Gate (80% Unanimous)"),
    )

    notree_base = [m[2]["direct_llm_no_tree"]["speed_200_records"]["baseline_direct_5vote_sec"] for m in loaded]
    tree_seq5 = [m[2]["tree_guided_50trees"]["speed_200_records"]["baseline_tree_seq_5vote_sec"] for m in loaded]
    save_dual_bar_plot(
        "results/fig_speedup_with_without_tree.png", labels,
        dict(
            series=[
                (notree_base, "Baseline (1 Student/Prompt, 5-Vote)", "#e63946"),
                ([m[2]["direct_llm_no_tree"]["speed_200_records"]["fast_packed_direct_5vote_sec"] for m in loaded], "Packed (5 Students/Prompt, 5-Vote)", "#457b9d"),
                ([m[2]["direct_llm_no_tree"]["speed_200_records"]["fast_packed_direct_1vote_sec"] for m in loaded], "Packed (5 Students/Prompt, 1-Vote)", "#2a9d8f"),
            ],
            ylabel="Latency for N=200 Students (sec, lower is faster)", title="Speedup WITHOUT Tree: Direct Classification Latency",
            ylim=(0, max(notree_base) * 1.18), fmt="{:.2f}s", y_pad=0.03, fontsize=7.0, legend_loc="upper left", bar_width=0.25,
        ),
        dict(
            series=[
                (tree_seq5, "Baseline Seq (1Q/Prompt, 5-Vote)", "#e76f51"),
                ([m[2]["tree_guided_50trees"]["speed_200_records"]["fast_tree_packed_5vote_sec"] for m in loaded], "Packed (3Q/Prompt, 5-Vote)", "#457b9d"),
                ([m[2]["tree_guided_50trees"]["speed_200_records"]["fast_tree_packed_3vote_sec"] for m in loaded], "Packed (3Q/Prompt, 3-Vote)", "#2a9d8f"),
                ([m[2]["tree_guided_50trees"]["speed_200_records"]["fast_tree_routed_1vote_sec"] for m in loaded], "Branch-Routed (1-Vote)", "#f4a261"),
                ([m[2]["tree_guided_50trees"]["speed_200_records"]["fast_tree_packed_1vote_sec"] for m in loaded], "Packed (3Q/Prompt, 1-Vote)", "#1d3557"),
            ],
            ylabel="Latency for N=200 Students (sec, lower is faster)", title="Speedup WITH Tree: Tree-Guided Inference Latency",
            ylim=(0, max(tree_seq5) * 1.18), fmt="{:.2f}s", y_pad=0.03, fontsize=6.2, legend_loc="upper left", bar_width=0.17,
        ),
    )
    print("Successfully generated summary_50seeds.json and all pre-meeting figures!")


if __name__ == "__main__":
    main()
