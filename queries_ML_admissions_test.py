import os
import pandas as pd
import numpy as np
import re
import json
import random
import yaml
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoProcessor, Llama4ForConditionalGeneration

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load config
with open("data/admissions/config.yaml") as f:
    config = yaml.safe_load(f)

# Load train data
with open("data/admissions/admission_train.json") as f:
    data = json.load(f)

# Assuming all feature lists are the same length
n_samples = len(data["math"])
records = []
for i in range(n_samples):
    record = {key: data[key][i] for key in data.keys()}
    # Create observation text using template (excluding label for fair evaluation)
    obs_template = """Here's a student's info:
        Math grade: ${math}
        Number of publications: ${num_publications}
        Number of recommendation letters: ${num_letters}
        Number of extracurricular activities: ${num_activities}
        Science grade: ${science}
        Art grade: ${art}
        Community service hours: ${community_service_hours}
        Sports participation: ${sports_participation}
        """
    observation = obs_template.replace("${math}", str(record["math"])) \
                              .replace("${num_publications}", str(record["num_publications"])) \
                              .replace("${num_letters}", str(record["num_letters"])) \
                              .replace("${num_activities}", str(record["num_activities"])) \
                              .replace("${science}", str(record["science"])) \
                              .replace("${art}", str(record["art"])) \
                              .replace("${community_service_hours}", str(record["community_service_hours"])) \
                              .replace("${sports_participation}", str(record["sports_participation"]))
    label = 1 if record["admission_distractor_10"] == "admitted" else 0
    records.append({"id": i, "text": observation, "target": label})

# Load test data
with open("data/admissions/admission_test.json") as f:
    test_data = json.load(f)

test_records = []
for i in range(len(test_data["math"])):
    record = {key: test_data[key][i] for key in test_data.keys()}
    # Create observation text using template (excluding label for fair evaluation)
    obs_template = """Here's a student's info:
        Math grade: ${math}
        Number of publications: ${num_publications}
        Number of recommendation letters: ${num_letters}
        Number of extracurricular activities: ${num_activities}
        Science grade: ${science}
        Art grade: ${art}
        Community service hours: ${community_service_hours}
        Sports participation: ${sports_participation}
        """
    observation = obs_template.replace("${math}", str(record["math"])) \
                              .replace("${num_publications}", str(record["num_publications"])) \
                              .replace("${num_letters}", str(record["num_letters"])) \
                              .replace("${num_activities}", str(record["num_activities"])) \
                              .replace("${science}", str(record["science"])) \
                              .replace("${art}", str(record["art"])) \
                              .replace("${community_service_hours}", str(record["community_service_hours"])) \
                              .replace("${sports_participation}", str(record["sports_participation"]))
    label = 1 if record["admission_distractor_10"] == "admitted" else 0
    test_records.append({"id": i, "text": observation, "target": label})

df = pd.DataFrame(records)
test_df = pd.DataFrame(test_records)


MODEL_PATH = "/scratch/ic2664/LLMs/Llama-4-Scout-17B-16E-Instruct"

print("Loading processor/tokenizer...")
try:
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer = processor.tokenizer
except Exception:
    processor = None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model...")
model = Llama4ForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()
print("Model loaded.")


def generate_text(prompt, max_new_tokens=512, temperature=0.3):
    """Generation wrapper compatible with any HuggingFace chat model."""
    if processor:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    else:
        messages = [{"role": "user", "content": prompt}]
    proc = processor if processor else tokenizer
    inputs = proc.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )
    return proc.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]


def build_query_generation_prompt(pos_texts, neg_texts, n_queries=5):
    """
    Build a prompt that asks the LLM to generate yes/no presence queries
    that distinguish admitted from non-admitted students in the admissions dataset.
    """
    pos_block = "\n---\n".join([f"Student {i+1}: {t[:500]}" for i, t in enumerate(pos_texts)])
    neg_block = "\n---\n".join([f"Student {i+1}: {t[:500]}" for i, t in enumerate(neg_texts)])

    prompt = f"""You are a university admissions officer. Your task is to generate yes/no questions that distinguish two groups of students based on their described profiles.

    Rules:
    1. Only ask about attributes EXPLICITLY STATED in the student profiles.
    2. Do NOT ask questions about admission directly — that is the label you are trying to predict.
    3. Each question should have DIFFERENT answers for Group A vs Group B.
    4. Each question must cover a DIFFERENT attribute or attribute value — no redundancy.

    Here is an example of how to do this:

    EXAMPLE GROUP A (admitted students):
    Student 1: Student has A in math, 5 publications, 3 recommendation letters, 4 extracurricular activities, A in science, B in art, 20 community service hours, and participates in 2 sports.
    Student 2: Student has B in math, 2 publications, 2 recommendation letters, 3 extracurricular activities, A in science, A in art, 15 community service hours, and participates in 1 sport.

    EXAMPLE GROUP B (non-admitted students):
    Student 1: Student has C in math, 0 publications, 1 recommendation letter, 1 extracurricular activity, C in science, D in art, 5 community service hours, and participates in 0 sports.
    Student 2: Student has D in math, 1 publication, 1 recommendation letter, 2 extracurricular activities, B in science, C in art, 10 community service hours, and participates in 1 sport.

    Good questions:
    - Does the student have an A in math? (Group A: Yes, No. Group B: No, No — discriminative)
    - Does the student have more than 2 publications? (Group A: Yes, No. Group B: No, No — discriminative)

    Bad questions:
    - Does the student have 3 recommendation letters? (Group A: Yes, No. Group B: No, No — NOT discriminative)
    - Does the student have an A in science? (Group A: Yes, Yes. Group B: No, No — NOT discriminative)
    - Is the student admitted? (This is the label itself — NOT allowed)

    Now do the same for these real student profiles:

    GROUP A (admitted students):
    {pos_block}

    GROUP B (non-admitted students):
    {neg_block}

    Generate exactly {n_queries} yes/no questions where Group A and Group B would have DIFFERENT answers.

    1."""
    return prompt

def parse_queries(raw_text, expected_n=5):
    """
    Parse numbered queries from LLM output.
    """
    queries = []
    
    # Match numbered lines: '1. ...', '2) ...', '1- ...'
    pattern = r'^\s*\d+\s*[.):\-]\s*(.+)'
    matches = re.findall(pattern, raw_text, re.MULTILINE)
    
    if matches:
        queries = [m.strip() for m in matches]
    
    # Clean up: ensure each ends with '?' and has reasonable length
    cleaned = []
    for q in queries:
        q = q.rstrip('?').strip()
        if len(q) > 10 and len(q) < 200:
            cleaned.append(q + '?')
    
    return cleaned[:expected_n]



def generate_queries(pos_texts, neg_texts, n_queries=5, debug=True):
    """
    Generate discriminative yes/no queries.
    Uses a single LLM call since Llama-4 follows instructions well.
    """
    prompt = build_query_generation_prompt(pos_texts, neg_texts, n_queries)
    raw = generate_text(prompt, max_new_tokens=512, temperature=0.4)
    
    if debug:
        print(f"  Raw output:\n{raw[:600]}")
    
    queries = parse_queries(raw, expected_n=n_queries)
    
    if debug:
        print(f"  Parsed {len(queries)}/{n_queries} queries")
    
    # If we got fewer than expected, retry with higher temperature
    if len(queries) < n_queries:
        raw2 = generate_text(prompt, max_new_tokens=512, temperature=0.7)
        queries2 = parse_queries(raw2, expected_n=n_queries)
        if len(queries2) > len(queries):
            queries = queries2
    
    return queries[:n_queries]

def build_answer_prompt(report_text, query):
    prompt = f"""Read the following student profile and answer the question.

    Student profile:
    {report_text[:800]}

    Question: {query}

    First, quote the most relevant part of the student profile. Then answer with exactly "Yes" or "No".
    """
    return prompt


def answer_query(report_text, query):
    """Returns 1 for Yes, 0 for No, np.nan if unclear."""
    prompt = build_answer_prompt(report_text, query)
    raw = generate_text(prompt, max_new_tokens=100, temperature=0.1)
    raw_lower = raw.strip().lower()

    last_yes = raw_lower.rfind('yes')
    last_no = raw_lower.rfind('no')

    if last_yes == -1 and last_no == -1:
        return np.nan
    elif last_yes == -1:
        return 0
    elif last_no == -1:
        return 1
    else:
        return 1 if last_yes > last_no else 0


def answer_queries_batch(texts, queries, desc="Answering queries"):
    """Returns DataFrame of shape (n_texts, n_queries) with binary values."""
    results = []
    for text in tqdm(texts, desc=desc):
        row = [answer_query(text, q) for q in queries]
        results.append(row)
    feature_df = pd.DataFrame(results, columns=[f"Q{i}" for i in range(len(queries))])
    feature_df = feature_df.fillna(0).astype(int)
    return feature_df


# ============================================================
# MULTI-RUN EXPERIMENT
# ============================================================

N_RUNS = 5
N_QUERIES = 5
N_FEW_SHOT = 2  # pos/neg examples shown to LLM for query generation
START_RUN = int(os.environ.get("START_RUN", 1))

RESULTS_DIR = "results_admission"
os.makedirs(RESULTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

run_summary = []

for run_idx in range(START_RUN - 1, N_RUNS):
    run_seed = SEED + run_idx
    print(f"\n{'='*60}")
    print(f"RUN {run_idx + 1}/{N_RUNS}  (seed={run_seed})")
    print(f"{'='*60}")

    run_dir = os.path.join(RESULTS_DIR, f"run_{run_idx + 1:02d}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Sample fresh pos/neg examples each run
    pos = df[df['target'] == 1].sample(n=N_FEW_SHOT, random_state=run_seed)
    neg = df[df['target'] == 0].sample(n=N_FEW_SHOT, random_state=run_seed)

    few_shot_pos = pos['text'].tolist()
    few_shot_neg = neg['text'].tolist()

    # Generate queries
    queries = generate_queries(few_shot_pos, few_shot_neg, n_queries=N_QUERIES)
    print(f"\nGenerated {len(queries)} queries:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")

    # Save queries
    with open(os.path.join(run_dir, "queries.json"), "w") as f:
        json.dump({"run": run_idx + 1, "seed": run_seed, "queries": queries}, f, indent=2)

    # Evaluate queries on test dataset
    test_texts = test_df['text'].tolist()
    test_labels = test_df['target'].tolist()
    print(f"\nEvaluating on test dataset ({len(test_texts)} samples)...")
    feature_df = answer_queries_batch(test_texts, queries, desc=f"Run {run_idx + 1} test evaluation")
    feature_df.insert(0, "target", test_labels)

    # Save feature matrix
    features_path = os.path.join(run_dir, "features.csv")
    feature_df.to_csv(features_path, index=False)
    print(f"Feature matrix saved: {features_path}")

    # Per-query agreement with target (how discriminative each query is)
    query_stats = []
    for qi, q in enumerate(queries):
        col = f"Q{qi}"
        pos_rate = feature_df.loc[feature_df['target'] == 1, col].mean()
        neg_rate = feature_df.loc[feature_df['target'] == 0, col].mean()
        query_stats.append({
            "query_idx": qi,
            "query": q,
            "yes_rate_positive_class": round(float(pos_rate), 3),
            "yes_rate_negative_class": round(float(neg_rate), 3),
            "discrimination": round(abs(float(pos_rate) - float(neg_rate)), 3),
        })
        print(f"  Q{qi}: pos_yes={pos_rate:.2f}, neg_yes={neg_rate:.2f}, disc={abs(pos_rate-neg_rate):.2f}  |  {q}")

    with open(os.path.join(run_dir, "query_stats.json"), "w") as f:
        json.dump(query_stats, f, indent=2)

    nan_rate = feature_df.drop(columns=["target"]).isnull().mean().mean()
    run_summary.append({
        "run": run_idx + 1,
        "seed": run_seed,
        "n_queries_generated": len(queries),
        "avg_discrimination": round(float(np.mean([s["discrimination"] for s in query_stats])), 3),
        "nan_rate": round(float(nan_rate), 3),
        "results_dir": run_dir,
    })

# Save overall summary
summary_df = pd.DataFrame(run_summary)
summary_path = os.path.join(RESULTS_DIR, f"summary_{timestamp}.csv")
summary_df.to_csv(summary_path, index=False)

print(f"\n{'='*60}")
print(f"ALL RUNS COMPLETE")
print(f"Summary saved to: {summary_path}")
print(summary_df.to_string(index=False))