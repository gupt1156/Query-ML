"""
Reliability and Ground-Truth Fidelity Evaluator for Query-ML.

Measures two essential dimensions of reliability:
1. Ground-Truth Fidelity: % of time LLM extraction matches true synthetic records.
2. Consistency Across Runs: How often the LLM gives identical answers across positive temperature draws.
"""

import os
import sys
import re
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import numpy as np
import pandas as pd

from src.dataset import load_admissions_data
from src.generator import FastVLLMEngine


class GroundTruthParser:
    """Deterministic parser extracting ground-truth boolean answers from structured admissions records."""

    @staticmethod
    def evaluate_query(query: str, raw_record: Dict[str, Any]) -> int:
        q = query.lower().strip()

        # 1. Math Grade
        if "math" in q or "mathematics" in q:
            val = str(raw_record.get("math", "")).upper()
            if any(term in q for term in ["grade of a", "is a", "is an a", "received an a", "'a'", "grade a", "math grade is a", "math grade of a"]):
                return int(val == "A")
            elif any(term in q for term in ["grade of b", "is b", "received a b", "'b'", "grade b"]):
                return int(val == "B")
            elif "f grade" in q or "grade of f" in q:
                return int(val == "F")
            elif "at least a 'b'" in q or "at least b" in q:
                return int(val in ["A", "B"])

        # 2. Number of Publications
        if "publication" in q or "paper" in q or "published" in q:
            val = int(raw_record.get("num_publications", 0))
            m = re.search(r"at least (\d+)|(\d+)\s*(?:or more|\+)", q)
            thresh = int(m.group(1) or m.group(2)) if m else 1
            return int(val >= thresh)

        # 3. Extracurricular Activities
        if "extracurricular" in q or "activities" in q or "activity" in q:
            val = int(raw_record.get("num_activities", 0))
            m = re.search(r"at least (\d+)|(\d+)\s*(?:or more|\+)", q)
            thresh = int(m.group(1) or m.group(2)) if m else 2
            return int(val >= thresh)

        # 4. Recommendation Letters
        if "letter" in q or "recommendation" in q:
            val = int(raw_record.get("num_letters", 0))
            m = re.search(r"at least (\d+)|(\d+)\s*(?:or more|\+)", q)
            thresh = int(m.group(1) or m.group(2)) if m else 1
            return int(val >= thresh)

        # 5. Community Service Hours
        if "community service" in q:
            val = float(raw_record.get("community_service_hours", 0))
            m = re.search(r"at least (\d+)|(\d+)\s*(?:or more|\+)", q)
            thresh = float(m.group(1) or m.group(2)) if m else 20.0
            return int(val >= thresh)

        # 6. Science Grade
        if "science" in q:
            val = str(raw_record.get("science", "")).upper()
            if any(term in q for term in ["grade of a", "is an a", "grade a"]):
                return int(val == "A")
            elif any(term in q for term in ["grade of b", "is a b", "grade b"]):
                return int(val == "B")
            elif "grade of f" in q or "is an f" in q:
                return int(val == "F")
            elif "c or lower" in q:
                return int(val in ["C", "D", "F"])

        # 7. Art Grade
        if "art" in q:
            val = str(raw_record.get("art", "")).upper()
            if any(term in q for term in ["grade of a", "is an a", "grade a"]):
                return int(val == "A")

        # 8. Sports Participation
        if "sport" in q or "athlet" in q:
            val = int(raw_record.get("sports_participation", 0))
            m = re.search(r"at least (\d+)|(\d+)\s*(?:or more|\+)", q)
            thresh = int(m.group(1) or m.group(2)) if m else 1
            return int(val >= thresh)

        return -1


def categorize_query(query: str) -> str:
    txt = query.lower()
    if "math" in txt:
        return "true_causal_math"
    elif "publication" in txt or "paper" in txt:
        return "true_causal_publications"
    elif "activit" in txt or "extracurricular" in txt:
        return "true_causal_activities"
    elif "letter" in txt:
        return "distractor_letters"
    elif "service" in txt or "hours" in txt:
        return "distractor_community_service"
    elif "science" in txt:
        return "distractor_science"
    elif "art" in txt:
        return "distractor_art"
    elif "sport" in txt:
        return "distractor_sports"
    return "other_unknown"


def compute_side_reliability(
    queries: List[str],
    raw_records: List[Dict[str, Any]],
    engine: FastVLLMEngine,
    cache_prefix: str = "test",
    majority_votes: int = 5
) -> Dict[str, Any]:
    """
    Computes Ground-Truth Fidelity and Consistency Across Runs for a list of queries on the fly.
    Uses cached inference votes from the FastVLLMEngine.
    """
    if not queries or not raw_records:
        return {}

    prefix = f"{cache_prefix}_maj{majority_votes}" if majority_votes > 1 else cache_prefix
    parser = GroundTruthParser()

    query_reports = []
    by_cat = defaultdict(list)

    for q in queries:
        gt_vals = np.array([parser.evaluate_query(q, r) for r in raw_records])
        if -1 in gt_vals:
            # Not verifiable by ground truth parser
            continue

        cache_key = f"{prefix}::{q}"
        if cache_key not in engine.query_cache:
            continue

        maj_preds = engine.query_cache[cache_key]
        raw_votes = engine.raw_votes_cache.get(cache_key) # (N, votes)

        if raw_votes is None:
            raw_votes = maj_preds[:, None]

        # 1. Ground Truth Fidelity (Majority Vote)
        fidelity_maj = float((maj_preds == gt_vals).mean())

        # 2. Ground Truth Fidelity (Single Stochastic Draw)
        fidelity_single = float((raw_votes == gt_vals[:, None]).mean())

        # 3. Flip Rate (eta)
        flip_rate = float(1.0 - fidelity_single)

        # 4. Consistency Across Runs: mean record agreement max(p, 1-p)
        mean_p = raw_votes.mean(axis=1)
        record_agreement = np.maximum(mean_p, 1.0 - mean_p)
        consistency = float(record_agreement.mean())

        cat = categorize_query(q)
        rep = {
            "query": q,
            "category": cat,
            "ground_truth_fidelity_majority": round(fidelity_maj, 4),
            "ground_truth_fidelity_single_draw": round(fidelity_single, 4),
            "consistency_across_runs": round(consistency, 4),
            "flip_rate_eta": round(flip_rate, 4),
            "majority_voting_gain": round(fidelity_maj - fidelity_single, 4)
        }
        query_reports.append(rep)
        by_cat[cat].append(rep)

    if not query_reports:
        return {"num_verifiable_queries": 0}

    cat_summaries = {}
    for cat, items in by_cat.items():
        cat_summaries[cat] = {
            "count": len(items),
            "fidelity_majority_pct": round(float(np.mean([x["ground_truth_fidelity_majority"] for x in items])) * 100, 2),
            "fidelity_single_draw_pct": round(float(np.mean([x["ground_truth_fidelity_single_draw"] for x in items])) * 100, 2),
            "consistency_across_runs_pct": round(float(np.mean([x["consistency_across_runs"] for x in items])) * 100, 2),
            "flip_rate_pct": round(float(np.mean([x["flip_rate_eta"] for x in items])) * 100, 2),
            "majority_gain_pct": round(float(np.mean([x["majority_voting_gain"] for x in items])) * 100, 2)
        }

    return {
        "num_total_queries": len(queries),
        "num_verifiable_queries": len(query_reports),
        "fidelity_majority_pct": round(float(np.mean([x["ground_truth_fidelity_majority"] for x in query_reports])) * 100, 2),
        "fidelity_single_draw_pct": round(float(np.mean([x["ground_truth_fidelity_single_draw"] for x in query_reports])) * 100, 2),
        "consistency_across_runs_pct": round(float(np.mean([x["consistency_across_runs"] for x in query_reports])) * 100, 2),
        "flip_rate_pct": round(float(np.mean([x["flip_rate_eta"] for x in query_reports])) * 100, 2),
        "majority_gain_pct": round(float(np.mean([x["majority_voting_gain"] for x in query_reports])) * 100, 2),
        "by_category": cat_summaries,
        "query_details": query_reports
    }
