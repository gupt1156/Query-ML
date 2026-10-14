"""
Shared Utilities:
1. Domain-Agnostic Query Parsing, Normalization, Sanitization, Vote Consistency & Tree Traversal.
2. Benchmark-Specific Dataset Loader & Ground-Truth Evaluation Metrics.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

STOPWORDS = {"a", "an", "the", "of", "in", "for", "with", "to", "on", "at", "by", "record", "info", "item", "entry", "number", "count", "total", "value"}


def _canonicalize_attribute_phrase(attr_raw: str) -> str:
    """Normalizes an arbitrary attribute noun phrase into a canonical token key (singularized, stopwords removed)."""
    words = re.sub(r"[^\w\s]", " ", attr_raw.lower()).split()
    tokens = []
    for w in words:
        if w in STOPWORDS:
            continue
        if w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "y"
        elif w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
            w = w[:-1]
        tokens.append(w)
    return "_".join(tokens) if tokens else attr_raw.lower().strip()


def parse_query_semantics(query: str) -> Optional[Tuple[str, str, Any]]:
    """
    Domain-agnostic semantic parser for atomic natural language queries.
    Extracts (canonical_attribute, operator, threshold_or_value) from any atomic query,
    or returns None if the query is compound or uses vague ordinal comparisons.
    """
    q = query.lower().strip().rstrip("?.!")
    if re.search(r"\b(and|or|either|neither|both|nor|higher|lower|below|above|better|worse|fewer|less|at most)\b", q):
        return None

    # 1. Numerical lower-bound: "at least X <attr>" or "more/greater than X <attr>"
    m_ge = re.search(r"\bat least\s+(\d+)\s+([a-z][a-z0-9\s_\-]*)$", q)
    if m_ge and " of " not in m_ge.group(2):
        val, attr = int(m_ge.group(1)), _canonicalize_attribute_phrase(m_ge.group(2))
        return (attr, ">=", val) if (val > 0 and attr) else None

    m_gt = re.search(r"\b(?:more|greater)\s+than\s+(\d+)\s+([a-z][a-z0-9\s_\-]*)$", q)
    if m_gt and " of " not in m_gt.group(2):
        val, attr = int(m_gt.group(1)) + 1, _canonicalize_attribute_phrase(m_gt.group(2))
        return (attr, ">=", val) if (val > 0 and attr) else None

    # 2. Categorical / exact value equality:
    #    e.g. "Does the record show a <attr> of <val>?" or "Is the <attr> <val>?"
    for pat in (
        r"\b(?:show|shows|have|has|indicate|indicates|list|lists|mention|mentions|display|displays|state|states|with|is)\s+(?:an?\s+|the\s+)?([a-z][a-z0-9\s_\-]{1,40}?)\s+(?:of|is|as|equal\s+to)\s+(?:an?\s+)?['\"]?([a-z0-9_\-]+)['\"]?$",
        r"\b(?:show|shows|have|has|indicate|indicates|list|lists)\s+an?\s+['\"]?([a-z0-9]+)['\"]?\s+in\s+([a-z][a-z0-9\s_\-]{1,30})$",
    ):
        m_eq = re.search(pat, q)
        if m_eq:
            attr_raw, val_str = (m_eq.group(2), m_eq.group(1)) if " in " in pat else (m_eq.group(1), m_eq.group(2))
            attr = _canonicalize_attribute_phrase(attr_raw)
            if attr and not val_str.isdigit():
                return (attr, "==", val_str.upper())

    return None


def normalize_query_for_dedup(query: str) -> str:
    """Maps a query to a domain-agnostic canonical semantic key so paraphrases deduplicate cleanly."""
    sem = parse_query_semantics(query)
    if sem is not None:
        return f"{sem[0]}::{sem[1]}::{sem[2]}"
    q = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", query.lower().strip()))
    for verb in ("indicate", "list", "mention", "display", "have"):
        q = q.replace(verb, "show")
    return q


def sanitize_query_line(raw_line: str) -> Optional[str]:
    """Domain-agnostic cleaner for LLM-generated question lines enforcing atomic phrasing."""
    cleaned = re.sub(r"^\s*(?:\d+[\.\)\:-]|[\*\-\•])\s*", "", raw_line.strip()).strip()
    if not cleaned:
        return None
    for q_char in ('"', "'", "`", "“", "”"):
        cleaned = cleaned.strip(q_char).strip()
    cleaned = (cleaned.split("?")[0].strip() + "?") if "?" in cleaned else (cleaned + "?")

    for pat, repl in (
        (r"\bat least (\d+) number of\b", r"at least \1"),
        (r"\b(\d+) number of\b", r"\1"),
        (r"\bat least (\d+)\s+or more\b", r"at least \1"),
        (r"\b(\d+)\s+or more\b", r"at least \1"),
        (r"\bat least 1\s+([a-z][a-z\s]{1,25}?)\s+of\s+([a-z0-9]+)\b", r"a \1 of \2"),
        (r"\b1 ([a-z]{3,})ies\b", r"1 \1y"),
        (r"\b1 ([a-z]{3,}[^s])s\b", r"1 \1"),
    ):
        cleaned = re.sub(pat, repl, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if (
        not cleaned.lower().startswith(("does", "is", "has", "did", "are", "was", "were", "can"))
        or len(cleaned.split()) < 4
        or len(cleaned) < 15
        or parse_query_semantics(cleaned) is None
    ):
        return None
    return cleaned


def extract_valid_queries(raw_llm_output: str, max_queries: int = 5) -> List[str]:
    extracted, seen_norms = [], set()
    for line in raw_llm_output.strip().split("\n"):
        sanitized = sanitize_query_line(line)
        if sanitized is not None:
            norm = normalize_query_for_dedup(sanitized)
            if norm not in seen_norms:
                seen_norms.add(norm)
                extracted.append(sanitized)
                if len(extracted) >= max_queries:
                    break
    return extracted


def extract_answers_from_llm_response(raw_llm_output: str, num_queries: int) -> List[int]:
    answers = [0] * num_queries
    matched_numbered, unnumbered = False, []
    for line in raw_llm_output.strip().split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        matches = re.findall(r"(\d+)[\.\)\:\-]?\s*[\*_]*(Yes|No)\b", stripped, re.IGNORECASE)
        if matches:
            for num_str, yn_str in matches:
                idx = int(num_str) - 1
                if 0 <= idx < num_queries:
                    answers[idx] = int(yn_str.lower() == "yes")
                    matched_numbered = True
            continue
        if num_queries == 1:
            m_single = re.search(r"\b(Yes|No)\b", stripped, re.IGNORECASE)
            if m_single:
                return [int(m_single.group(1).lower() == "yes")]
        m_unnum = re.search(r"^\s*[\*\-\•]?\s*(Yes|No)\b", stripped, re.IGNORECASE)
        if m_unnum:
            unnumbered.append(int(m_unnum.group(1).lower() == "yes"))

    if not matched_numbered and unnumbered:
        for idx, val in enumerate(unnumbered[:num_queries]):
            answers[idx] = val
    return answers


def compute_vote_consistency(votes_matrix: Optional[np.ndarray], pairwise: bool = False) -> Tuple[float, float, float]:
    """Returns (consistency, unanimous_frac, high_agree_frac) from an (N, K) binary vote matrix."""
    if votes_matrix is None or votes_matrix.ndim != 2 or votes_matrix.shape[1] <= 1 or len(votes_matrix) == 0:
        return 1.0, 1.0, 1.0

    K = votes_matrix.shape[1]
    pos_votes = votes_matrix.sum(axis=1)
    agreement = np.maximum(pos_votes, K - pos_votes) / float(K)
    high_threshold = 0.799 if K >= 5 else (0.66 if K >= 3 else 0.999)
    unanimous, high_agree = float(np.mean(agreement >= 0.999)), float(np.mean(agreement >= high_threshold))
    if pairwise:
        pw = (pos_votes * (pos_votes - 1) + (K - pos_votes) * (K - pos_votes - 1)) / float(K * (K - 1))
        return float(np.mean(pw)), unanimous, high_agree
    return float(np.mean(agreement)), unanimous, high_agree


def traverse_mermaid_tree(mermaid_str: str, q_ans_dict: Dict[str, np.ndarray], n_samples: int) -> np.ndarray:
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: Dict[Tuple[str, int], str] = {}
    for line in mermaid_str.splitlines():
        if '["Class:' in line:
            nodes[line.split("[")[0].strip()] = {"is_leaf": True, "pred": int(line.split("Class: ")[1].split("<")[0].strip())}
        elif '["[' in line:
            nodes[line.split("[")[0].strip()] = {"is_leaf": False, "query": line.split("] ")[1].split("<br/>")[0].strip()}
        elif "-->|" in line:
            u, rest = line.split("-->|")[0].strip(), line.split("-->|")[1].split("|")
            edges[(u, 1 if rest[0].strip() == "Yes" else 0)] = rest[1].strip()

    preds = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        curr = "N0"
        while curr in nodes and not nodes[curr]["is_leaf"]:
            nxt = edges.get((curr, int(q_ans_dict[nodes[curr]["query"]][i])))
            if nxt is None or nxt not in nodes:
                break
            curr = nxt
        preds[i] = nodes.get(curr, {}).get("pred", 0)
    return preds


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if len(y_true) > 0 else 0.0


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_t, y_s = np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float)
    pos, neg = y_s[y_t == 1], y_s[y_t == 0]
    if len(pos) == 0 or len(neg) == 0:
        return accuracy_score(y_t, (y_s >= 0.5).astype(int))
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (len(pos) * len(neg)))


# ==============================================================================
# Benchmark-Specific Dataset & Ground-Truth Evaluation Helpers (Admissions)
# ==============================================================================

RECORD_TEXT_TEMPLATE = """Here's a student's info:
Math grade: {math}
Number of publications: {num_publications}
Number of recommendation letters: {num_letters}
Number of extracurricular activities: {num_activities}
Science grade: {science}
Art grade: {art}
Community service hours: {community_service_hours}
Sports participation: {sports_participation}"""

FIELD_KEYS = (
    "math", "num_publications", "num_letters", "num_activities",
    "science", "art", "community_service_hours", "sports_participation",
)

FIELD_TO_RAW_KEY: Dict[str, str] = {
    "true_causal_math": "math",
    "distractor_science": "science",
    "distractor_art": "art",
    "true_causal_publications": "num_publications",
    "true_causal_activities": "num_activities",
    "distractor_letters": "num_letters",
    "distractor_community_service": "community_service_hours",
    "distractor_sports": "sports_participation",
}

CATEGORY_MAPPING_RULES: List[Tuple[Tuple[str, ...], str]] = [
    (("math", "mathematics"), "true_causal_math"),
    (("publication", "publications", "paper", "papers", "author", "publish", "published"), "true_causal_publications"),
    (("activity", "activities", "extracurricular", "extracurriculars"), "true_causal_activities"),
    (("science",), "distractor_science"),
    (("sport", "sports", "athletic", "athletics"), "distractor_sports"),
    (("art",), "distractor_art"),
    (("letter", "letters", "recommendation", "recommendations"), "distractor_letters"),
    (("service", "community", "hour", "hours"), "distractor_community_service"),
]

CAUSAL_CATEGORIES = {"true_causal_math", "true_causal_publications", "true_causal_activities"}
GRADE_CATEGORIES = ("true_causal_math", "distractor_science", "distractor_art")


def format_single_student_record(raw_record: Dict[str, Any]) -> str:
    return RECORD_TEXT_TEMPLATE.format(**{k: raw_record.get(k, "N/A") for k in FIELD_KEYS})


def format_raw_dataset(raw_data_dict: Dict[str, List[Any]]) -> pd.DataFrame:
    rows = []
    for idx in range(len(raw_data_dict.get("math", []))):
        raw_row = {col: raw_data_dict[col][idx] for col in raw_data_dict}
        has_math_a = str(raw_row.get("math", "")).strip().upper() == "A"
        target = int(
            (has_math_a and int(raw_row.get("num_publications", 0)) >= 1)
            or (not has_math_a and int(raw_row.get("num_activities", 0)) >= 2)
        )
        rows.append({"id": idx, "text": format_single_student_record(raw_row), "target": target, "raw": raw_row})
    return pd.DataFrame(rows)


def load_admissions_data(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "admissions")
    train_df = format_raw_dataset(json.load(open(os.path.join(data_dir, "admission_train.json"))))
    test_df = format_raw_dataset(json.load(open(os.path.join(data_dir, "admission_test.json"))))
    val_path = os.path.join(data_dir, "admission_val.json")
    val_df = format_raw_dataset(json.load(open(val_path))) if os.path.exists(val_path) else None
    return train_df, val_df, test_df


def classify_query(query: str) -> str:
    q_lower = query.lower()
    for triggers, cat in CATEGORY_MAPPING_RULES:
        if any(re.search(rf"\b{re.escape(t)}\b", q_lower) for t in triggers):
            return cat
    return "other_unknown"


def is_causal(category: str) -> bool:
    return category in CAUSAL_CATEGORIES


def evaluate_ground_truth_query(query: str, raw_record: Dict[str, Any]) -> int:
    q = query.lower().strip()
    if "math grade of a and at least 1 publication" in q:
        return int(str(raw_record.get("math", "")).upper() == "A" and int(raw_record.get("num_publications", 0)) >= 1)

    cat = classify_query(query)
    raw_key = FIELD_TO_RAW_KEY.get(cat)
    if raw_key is None:
        return -1

    sem = parse_query_semantics(query)
    if sem is not None:
        _, op, target = sem
        rec_val = raw_record.get(raw_key)
        if cat in GRADE_CATEGORIES:
            return int(str(rec_val).upper().strip() == str(target).upper().strip())
        if str(rec_val).lower() in ("yes", "no"):
            return int((1.0 if str(rec_val).lower() == "yes" else 0.0) >= float(target))
        return int(float(rec_val or 0) >= float(target)) if op == ">=" else int(float(rec_val or 0) == float(target))

    if cat in GRADE_CATEGORIES:
        val = str(raw_record.get(raw_key, "")).upper().strip()
        for phrases, allowed in (
            (("b or higher", "at least b", "above c"), ("A", "B")),
            (("c or higher", "at least c"), ("A", "B", "C")),
            (("d or lower", "below c", "worse than c"), ("D", "F")),
            (("c or lower", "below b"), ("C", "D", "F")),
            (("b or lower", "lower than a", "below a"), ("B", "C", "D", "F")),
        ):
            if any(p in q for p in phrases):
                return int(val in allowed)
        return -1

    num_val = float(raw_record.get(raw_key, 0) or 0)
    if "0 " in q or "no " in q:
        return int(num_val == 0)
    m_num = re.search(r"(\d+)", q)
    if m_num:
        target = float(m_num.group(1))
        if "at most" in q or "fewer" in q or "less" in q:
            return int(num_val <= target if ("at most" in q or "or " in q) else num_val < target)
        return int(num_val > target if ("more than" in q or "greater than" in q) else num_val >= target)
    return -1


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    splits: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    y_t, y_p = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    acc = accuracy_score(y_t, y_p)
    tp = float(np.sum((y_t == 1) & (y_p == 1)))
    fp = float(np.sum((y_t == 0) & (y_p == 1)))
    fn = float(np.sum((y_t == 1) & (y_p == 0)))
    f1 = (2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 0.0
    auc = roc_auc_score(y_t, y_prob) if (y_prob is not None and len(np.unique(y_t)) > 1) else acc

    splits_list = splits or []
    total_splits = len(splits_list)
    causal_splits = sum(1 for s in splits_list if is_causal(s.get("category", "")))
    causal_found = {s["category"] for s in splits_list if is_causal(s.get("category", ""))}

    return {
        "accuracy": acc,
        "roc_auc": auc,
        "f1": float(f1),
        "total_splits": total_splits,
        "causal_splits": causal_splits,
        "causal_split_pct": round((causal_splits / total_splits) * 100, 2) if total_splits > 0 else 0.0,
        "causal_recovery_rate": round((len(causal_found) / len(CAUSAL_CATEGORIES)) * 100, 2),
        "splits_analysis": splits_list,
    }


def compute_side_reliability(
    queries: List[str],
    raw_records: List[Dict[str, Any]],
    engine: Any,
    majority_votes: int = 5,
    pairwise: bool = False,
) -> Dict[str, Any]:
    if not queries or not raw_records:
        return {}

    texts = [format_single_student_record(r) for r in raw_records]
    reports: List[Dict[str, Any]] = []
    for q in queries:
        gt_vals = np.array([evaluate_ground_truth_query(q, r) for r in raw_records])
        raw_votes = engine.get_votes_matrix(texts, q, majority_votes=majority_votes)
        if -1 in gt_vals or raw_votes is None:
            continue
        maj_preds = (np.mean(raw_votes, axis=1) >= 0.5).astype(int)
        fidelity_maj = float((maj_preds == gt_vals).mean())
        fidelity_single = float((raw_votes == gt_vals[:, None]).mean())
        consistency, unanimous, high_agree = compute_vote_consistency(raw_votes, pairwise=pairwise)
        reports.append({
            "query": q,
            "category": classify_query(q),
            "fidelity_majority_pct": round(fidelity_maj * 100, 2),
            "fidelity_single_draw_pct": round(fidelity_single * 100, 2),
            "consistency_across_runs_pct": round(consistency * 100, 2),
            "unanimous_agreement_pct": round(unanimous * 100, 2),
            "high_agreement_pct": round(high_agree * 100, 2),
            "flip_rate_pct": round((1.0 - fidelity_single) * 100, 2),
        })

    if not reports:
        return {}
    keys = (
        "fidelity_majority_pct", "fidelity_single_draw_pct", "consistency_across_runs_pct",
        "unanimous_agreement_pct", "high_agreement_pct", "flip_rate_pct",
    )
    summary = {k: round(float(np.mean([r[k] for r in reports])), 2) for k in keys}
    summary["unanimous_pct"] = summary["unanimous_agreement_pct"]
    summary["query_details"] = reports
    return summary
