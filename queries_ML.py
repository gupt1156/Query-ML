import pandas as pd
import numpy as np
import re
import json
import random
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

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

df = pd.read_csv("/scratch/ic2664/Query-ML/data/mpg_simple_textual.txt")
df['text'] = df[df.columns[1:5]].astype(str).agg(" ".join, axis=1)
df = df[['text', 'target', 'id']]


# Balance classes and sample working subset
SEED = 41
N_PER_CLASS = 100  # 200 total — adjust based on your API budget

pos = df[df['target'] == 1].sample(n=N_PER_CLASS, random_state=SEED)
neg = df[df['target'] == 0].sample(n=N_PER_CLASS, random_state=SEED)
df_balanced = pd.concat([pos, neg]).sample(frac=1, random_state=SEED).reset_index(drop=True)


MODEL_PATH = "/scratch/ic2664/LLMs/Llama-4-Scout-17B-16E-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
print("Model loaded.")


def generate_text(prompt, max_new_tokens=512, temperature=0.3):
    """Generation wrapper using Llama 4 chat template."""
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def build_query_generation_prompt(pos_texts, neg_texts, n_queries=5):
    """
    Build a prompt that asks the LLM to generate yes/no presence queries
    that distinguish positive (good fuel efficiency) from negative (bad fuel efficiency)
    car descriptions in the MPG dataset.
    """
    pos_block = "\n---\n".join([f"Car {i+1}: {t[:500]}" for i, t in enumerate(pos_texts)])
    neg_block = "\n---\n".join([f"Car {i+1}: {t[:500]}" for i, t in enumerate(neg_texts)])

    prompt = f"""You are an automotive data analyst. Your task is to generate yes/no questions that distinguish two groups of cars based on their described attributes.

Each car is described using these attributes: number of cylinders, horsepower level, displacement level, weight, and acceleration.

Rules:
1. Only ask about attributes EXPLICITLY STATED in the car descriptions.
2. Do NOT ask questions about fuel efficiency directly — that is the label you are trying to predict.
3. Each question should have DIFFERENT answers for Group A vs Group B.
4. Each question must cover a DIFFERENT attribute or attribute value — no redundancy.

Here is an example of how to do this:

EXAMPLE GROUP A (good fuel efficiency):
Car 1: Car has 4 cylinders, low horsepower, low displacement, light weight, and fast acceleration.
Car 2: Car has 4 cylinders, low horsepower, light weight, moderate acceleration, and low displacement.

EXAMPLE GROUP B (bad fuel efficiency):
Car 1: Car has 8 cylinders, high horsepower, high displacement, heavy weight, and slow acceleration.
Car 2: Car has 6 cylinders, medium horsepower, average weight, medium displacement, and fast acceleration.

Good questions:
- Does the car have 4 cylinders? (Group A: Yes, Yes. Group B: No, No — discriminative)
- Does the car have high horsepower? (Group A: No, No. Group B: Yes, No — discriminative)

Bad questions:
- Does the car have fast acceleration? (Group A: Yes, No. Group B: No, Yes — NOT discriminative)
- Does the car have low displacement? (Group A: Yes, Yes. Group B: Yes, Yes — NOT discriminative)
- Does the car have good fuel efficiency? (This is the label itself — NOT allowed)


Now do the same for these real car descriptions:

GROUP A (good fuel efficiency):
{pos_block}

GROUP B (bad fuel efficiency):
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

# --- Test query generation with a small sample ---
sample_pos = pos['text'].sample(5, random_state=42).tolist()
sample_neg = neg['text'].sample(5, random_state=42).tolist()

test_queries = generate_queries(sample_pos, sample_neg, n_queries=5)
print("Generated queries:")
for i, q in enumerate(test_queries, 1):
    print(f"  {i}. {q}")

print("Positive samples:")
print(sample_pos)


print("Negative samples:")
print(sample_neg)

def build_answer_prompt(report_text, query):
    """
    Build a prompt that asks the LLM to answer a yes/no query
    based on a car description.
    """
    prompt = f"""Read the following car description and answer the question.

Car description:
{report_text[:800]}

Question: {query}

First, quote the most relevant part of the car description. Then answer with exactly "Yes" or "No".
"""
    return prompt


def answer_query(report_text, query):
    """
    Answer a single yes/no query for one report.
    Returns 1 for Yes, 0 for No, np.nan if unclear.
    """
    prompt = build_answer_prompt(report_text, query)
    raw = generate_text(prompt, max_new_tokens=100, temperature=0.1)
    raw_lower = raw.strip().lower()
    
    # Look for the final Yes/No (after the reasoning)
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
    """
    For each text, answer all queries. Returns a DataFrame of shape (n_texts, n_queries)
    with binary values (1=Yes, 0=No).
    """
    results = []
    for text in tqdm(texts, desc=desc):
        row = []
        for query in queries:
            ans = answer_query(text, query)
            row.append(ans)
        results.append(row)
    
    feature_df = pd.DataFrame(results, columns=[f"Q{i}" for i in range(len(queries))])
    feature_df = feature_df.fillna(0).astype(int)
    return feature_df

# --- Quick test: answer queries on a few reports ---
test_texts = sample_pos + sample_neg
test_features = answer_queries_batch(test_texts, test_queries, desc="Test answering")
print(test_features)
print(f"\nQuery labels: {test_queries}")