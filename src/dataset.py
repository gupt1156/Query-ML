"""
Dataset Loading, Schema Formatting, and Ground-Truth Causal Definitions.

This module handles:
1. Converting raw tabular student records into standardized natural language texts.
2. Ground-truth labeling for two distinct benchmark targets:
   - Benchmark 1 (Single-Track / Original): Admitted iff Math == 'A'
   - Benchmark 2 (Dual-Track / Contextual): Admitted iff (Math == 'A' and Pubs >= 1) or
                                                        (Math != 'A' and Activities >= 2)
3. Categorizing generated natural language queries into causal vs distractor mechanisms.
"""

import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

# Standard natural language template for converting tabular student profiles into textual records
RECORD_TEXT_TEMPLATE = """Here's a student's info:
Math grade: {math}
Number of publications: {num_publications}
Number of recommendation letters: {num_letters}
Number of extracurricular activities: {num_activities}
Science grade: {science}
Art grade: {art}
Community service hours: {community_service_hours}
Sports participation: {sports_participation}"""

# Mapping of query keywords/triggers to standardized feature categories
CATEGORY_MAPPING_RULES: List[Tuple[Any, str]] = [
    ("math", "true_causal_math"),
    (("publication", "paper", "author", "publish"), "true_causal_publications"),
    (("activity", "extracurricular", "activities"), "true_causal_activities"),
    ("science", "distractor_science"),
    ("art", "distractor_art"),
    (("letter", "recommendation"), "distractor_letters"),
    (("service", "community", "hours"), "distractor_community_service"),
    (("sport", "athlet"), "distractor_sports"),
]


def format_single_student_record(raw_record: Dict[str, Any]) -> str:
    """
    Formats a raw dictionary of student attributes into the standardized observation text.
    """
    field_keys = [
        "math",
        "num_publications",
        "num_letters",
        "num_activities",
        "science",
        "art",
        "community_service_hours",
        "sports_participation",
    ]
    template_values = {key: raw_record.get(key, "N/A") for key in field_keys}
    return RECORD_TEXT_TEMPLATE.format(**template_values)


def assign_ground_truth_target(raw_record: Dict[str, Any], task_type: str = "contextual") -> int:
    """
    Assigns the binary ground-truth target label based on benchmark rules:

    - Contextual (Dual-Track):
        Admitted (1) iff:
          (Math == 'A' AND Publications >= 1)  [Academic Track]
          OR
          (Math != 'A' AND Activities >= 2)    [Holistic Track]

    - Original (Single-Track):
        Admitted (1) iff Math == 'A' (mapped via 'admission_distractor_10' in raw data)
    """
    math_grade = str(raw_record.get("math", "")).strip().upper()
    has_math_a = (math_grade == "A")
    num_publications = int(raw_record.get("num_publications", 0))
    num_activities = int(raw_record.get("num_activities", 0))

    if task_type == "contextual":
        academic_track = has_math_a and (num_publications >= 1)
        holistic_track = (not has_math_a) and (num_activities >= 2)
        return 1 if (academic_track or holistic_track) else 0

    elif task_type == "conjunction":
        return 1 if (has_math_a and num_publications >= 1) else 0

    else:
        # Original single-track admissions rule
        orig_val = raw_record.get("admission_distractor_10", "")
        return 1 if orig_val in ("admitted", 1) else 0


def format_raw_dataset(raw_data_dict: Dict[str, List[Any]], task_type: str = "contextual") -> pd.DataFrame:
    """
    Converts raw column-oriented dictionary data into a structured pandas DataFrame.
    """
    num_records = len(raw_data_dict.get("math", []))
    formatted_rows: List[Dict[str, Any]] = []

    for idx in range(num_records):
        raw_row = {col_name: raw_data_dict[col_name][idx] for col_name in raw_data_dict}
        text_representation = format_single_student_record(raw_row)
        target_label = assign_ground_truth_target(raw_row, task_type=task_type)

        formatted_rows.append({
            "id": idx,
            "text": text_representation,
            "target": target_label,
            "raw": raw_row
        })

    return pd.DataFrame(formatted_rows)


def load_admissions_data(
    data_dir: Optional[str] = None,
    task_type: str = "contextual"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Loads training and test partitions for the admissions benchmark.

    Args:
        data_dir: Directory path containing 'admission_train.json' and 'admission_test.json'.
        task_type: Target benchmark regime ('contextual' or 'original').

    Returns:
        Tuple of (train_df, test_df, metadata_dict)
    """
    if data_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data", "admissions")

    train_file_path = os.path.join(data_dir, "admission_train.json")
    test_file_path = os.path.join(data_dir, "admission_test.json")

    with open(train_file_path, "r") as f_train:
        train_raw = json.load(f_train)

    with open(test_file_path, "r") as f_test:
        test_raw = json.load(f_test)

    train_df = format_raw_dataset(train_raw, task_type=task_type)
    test_df = format_raw_dataset(test_raw, task_type=task_type)

    metadata = {
        "num_train": len(train_df),
        "num_test": len(test_df),
        "task_type": task_type,
        "data_dir": data_dir
    }

    return train_df, test_df, metadata


def classify_query(query: str, task_type: str = "contextual") -> str:
    """
    Classifies a natural language query into a causal or distractor category based on keyword triggers.

    Args:
        query: Candidate question string (e.g. 'Does the record show a Math grade of A?').
        task_type: 'contextual' or 'original'.

    Returns:
        Standard category string (e.g. 'true_causal_math', 'distractor_community_service').
    """
    query_lower = query.lower()

    for triggers, category in CATEGORY_MAPPING_RULES:
        trigger_tuple = (triggers,) if isinstance(triggers, str) else triggers
        if any(trigger in query_lower for trigger in trigger_tuple):
            # In original benchmark, only Math is causal; publications and activities are distractors
            if task_type == "original" and category in ("true_causal_publications", "true_causal_activities"):
                return category.replace("true_causal_", "distractor_")
            return category

    return "other_unknown"


def is_causal(category: str, task_type: str = "contextual") -> bool:
    """
    Determines whether a given query category represents a true ground-truth causal mechanism.
    """
    if task_type == "original":
        ground_truth_causal_set: Set[str] = {"true_causal_math"}
    else:
        # Contextual benchmark includes Math, Publications, and Extracurricular Activities
        ground_truth_causal_set = {
            "true_causal_math",
            "true_causal_publications",
            "true_causal_activities"
        }

    return category in ground_truth_causal_set
