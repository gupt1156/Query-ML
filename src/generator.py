"""
Domain-Agnostic LLM Query Generation, Prompt Templates, and Answering Engine backed by vLLM.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import (
    extract_answers_from_llm_response,
    extract_valid_queries,
    normalize_query_for_dedup,
)

GENERIC_EXAMPLE_RECORD_1 = """Record info:
Category: A
Items: 0
Sessions: 2
Status: active"""

GENERIC_EXAMPLE_RECORD_2 = """Record info:
Category: B
Items: 1
Sessions: 1
Status: inactive"""


def build_query_generation_prompt(
    pos_texts: List[str],
    neg_texts: List[str],
    n_queries: int = 5,
    ancestor_queries: Optional[List[str]] = None,
) -> str:
    group_a = "\n---\n".join(f"Record A{i + 1}:\n{t.strip()}" for i, t in enumerate(pos_texts))
    group_b = "\n---\n".join(f"Record B{i + 1}:\n{t.strip()}" for i, t in enumerate(neg_texts))
    ancestor_constraint = ""
    if ancestor_queries:
        anc_str = "\n".join(f"- {q.strip()}" for q in ancestor_queries if q.strip())
        ancestor_constraint = f"""
---
ALREADY EVALUATED QUESTIONS IN THIS BRANCH:
{anc_str}

MANDATORY NON-REDUNDANCY RULE:
- Do NOT repeat or paraphrase any of the questions listed above.
- Prioritize NOVEL, UNTESTED attributes first.
- If you test a numerical attribute already mentioned above, you MUST use a strictly different, logically consistent threshold that further splits the remaining records.
"""

    return f"""You are an expert statistical machine learning researcher.

Your task is to identify simple, objective, verifiable yes/no questions that distinguish Group A from Group B based strictly on the text provided.

RULES:
1. STRICT FACTUAL PRESENCE: Only ask about concrete facts and exact values directly stated in the text. Queries must be self-contained and require zero outside knowledge to answer.
2. NO SUBJECTIVE INFERENCE: Do NOT ask subjective or predictive questions.
3. BIDIRECTIONAL CONTRASTIVE SEPARATION: A question is equally discriminative whether its answer is 'Yes' for Group A and 'No' for Group B, OR 'Yes' for Group B and 'No' for Group A. Test simple conditions shared by records in either Group A or Group B that distinguish them from the other group.
4. DIVERSITY: Each of the {n_queries} questions must focus on a distinct attribute or condition. Do NOT output duplicate or paraphrased questions.
5. ATOMIC SINGLE ATTRIBUTE ONLY: Each question must evaluate strictly ONE individual attribute. NEVER combine multiple attributes or values using 'and', 'or', or 'either'.
6. SELF-CONTAINED PHRASING (CATEGORICAL VS NUMERICAL):
   - For categorical or string attributes: ALWAYS test exact equality to a single specific value stated in the text (e.g., 'Does the record show a [Attribute] of [Value]?'). NEVER use count phrases ('at least 1 [Attribute] of [Value]') or ordinal inequalities ('or higher', 'or lower', 'below', 'above').
   - For numerical count or quantity attributes: ALWAYS use 'Does the record show at least X [Attribute]?' with a positive integer X >= 1. NEVER use 'at least 0', '0 or fewer', or 'at least X or more'.
7. MINIMAL SEPARATION THRESHOLDS: For numerical quantities, prefer the lowest clean integer threshold that separates the records (e.g., test 'at least 1' when distinguishing >= 1 from 0; test 'at least 2' when distinguishing >= 2 from <= 1).
{ancestor_constraint}
---
GROUP A:
{group_a}

---
GROUP B:
{group_b}

---
Generate exactly {n_queries} distinct, atomic yes/no questions.
Output numbered questions from 1 to {n_queries}, each on a separate line ending with a question mark.
"""


def build_multi_answer_prompt(record_text: str, queries: List[str]) -> str:
    numbered_queries = "\n".join(f"{i + 1}. {q.strip()}" for i, q in enumerate(queries))
    if len(queries) == 1:
        q_ex = "1. Does the record show at least 2 sessions?"
        ex1_ans, ex2_ans = "1. No", "1. Yes"
        rec1, rec2 = GENERIC_EXAMPLE_RECORD_2, GENERIC_EXAMPLE_RECORD_1
    else:
        q_ex = "1. Does the record show a Category of A?\n2. Does the record show at least 1 item?\n3. Does the record show at least 2 sessions?"
        ex1_ans, ex2_ans = "1. Yes\n2. No\n3. Yes", "1. No\n2. Yes\n3. No"
        rec1, rec2 = GENERIC_EXAMPLE_RECORD_1, GENERIC_EXAMPLE_RECORD_2

    return f"""Evaluate each yes/no question based strictly on the record provided below.

RULES:
1. STRICT FORMAT: Output ONLY the question number followed by 'Yes' or 'No' on a separate line (e.g. '1. Yes'). Do NOT output any explanation or extra words.
2. EXACT CATEGORICAL MATCH: For a categorical/value question (e.g., 'Does the record show a [Attribute] of [Value]?'), output 'Yes' ONLY if that exact attribute has that exact value; otherwise output 'No'.
3. NUMERICAL THRESHOLD RULE: 'at least X' or 'X or more' means the record's numerical value for that specific attribute is >= X. If the record's value is strictly less than X (e.g. 0 when asking 'at least 1', or 0 or 1 when asking 'at least 2'), you MUST output 'No'.

EXAMPLE 1:
Record:
{rec1}
Questions:
{q_ex}
Output:
{ex1_ans}

EXAMPLE 2:
Record:
{rec2}
Questions:
{q_ex}
Output:
{ex2_ans}

RECORD:
{record_text.strip()}

QUESTIONS:
{numbered_queries}

OUTPUT:"""


def build_direct_classification_prompt(pos_examples: str, neg_examples: str, target_records: List[str]) -> str:
    header = f"""EXAMPLES OF POSITIVE RECORDS (Class 1, Answer = Yes):
{pos_examples}

EXAMPLES OF NEGATIVE RECORDS (Class 0, Answer = No):
{neg_examples}

RULES:
1. Output ONLY the record number followed by 'Yes' (if Class 1) or 'No' (if Class 0), e.g. '1. Yes'.
2. Do NOT output any commentary or explanation."""

    if len(target_records) == 1:
        return f"Predict whether the target record below belongs to Class 1 ('Yes') or Class 0 ('No') based on the example records.\n\n{header}\n\nTARGET RECORD:\n{target_records[0].strip()}\n\nQUESTION:\n1. Does this record belong to Class 1?\n\nOUTPUT:"
    packed = "\n\n".join(f"Record {j + 1}:\n{t.strip()}" for j, t in enumerate(target_records))
    return f"Predict whether each target record below belongs to Class 1 ('Yes') or Class 0 ('No') based on the example records.\n\n{header}\n\nTARGET RECORDS:\n{packed}\n\nOUTPUT:"


def sample_contrastive_record_pairs(
    df: pd.DataFrame,
    num_subsets: int,
    samples_per_group: int = 2,
    base_seed: int = 42,
) -> List[Tuple[List[str], List[str]]]:
    pos_df, neg_df = df[df["target"] == 1], df[df["target"] == 0]
    if len(pos_df) == 0 or len(neg_df) == 0:
        return []
    n_pos, n_neg = min(samples_per_group, len(pos_df)), min(samples_per_group, len(neg_df))
    pairs = []
    for idx in range(num_subsets):
        s = base_seed + 10 * idx
        p, n = pos_df.sample(n=n_pos, random_state=s)["text"].tolist(), neg_df.sample(n=n_neg, random_state=s)["text"].tolist()
        pairs.append((p, n) if idx % 2 == 0 else (n, p))
    return pairs


class FastVLLMEngine:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", gpu_memory_utilization: float = 0.85):
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM
        print(f"Initializing FastVLLMEngine with model: {model_name}")
        kwargs = dict(
            model=model_name, tensor_parallel_size=1, gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=1024, trust_remote_code=True, enforce_eager=True,
        )
        try:
            self.llm = LLM(**kwargs, enable_prefix_caching=True)
        except TypeError:
            self.llm = LLM(**kwargs)
        self.record_vote_cache: Dict[Tuple[int, str], Dict[str, np.ndarray]] = {}
        self.default_chunk_size: int = 1 if "0.5b" in model_name.lower() else 8

    def generate_batch_text(
        self, prompts: List[str], max_tokens: int = 160, temperature: float = 0.3, n: int = 1, stop: Optional[List[str]] = None
    ) -> Any:
        if not prompts:
            return []
        from vllm import SamplingParams
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n, stop=stop or ["\n\n\n", "User:", "<|im_end|>"])
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outputs] if n == 1 else [[c.text.strip() for c in o.outputs] for o in outputs]

    def generate_queries_for_nodes(self, node_items: List[Dict[str, Any]], n_subsets: int = 8, k_queries: int = 5) -> None:
        all_prompts: List[str] = []
        node_slices: List[Tuple[Dict[str, Any], int, int, set]] = []

        for item in node_items:
            ancestors = [q for q, _ in item.get("branch_history", [])]
            all_excluded = set(ancestors)
            record_pairs = sample_contrastive_record_pairs(item["sub_df"], n_subsets, 2, item["seed"] + 100 * item.get("depth", 0))
            if not record_pairs:
                item["candidates"] = []
                continue
            start_idx = len(all_prompts)
            for pos_texts, neg_texts in record_pairs:
                all_prompts.append(build_query_generation_prompt(pos_texts, neg_texts, k_queries, ancestors))
            node_slices.append((item, start_idx, len(all_prompts), all_excluded))

        if not all_prompts:
            return
        raw_completions = self.generate_batch_text(all_prompts, max_tokens=180, temperature=0.3)
        for item, start_idx, end_idx, all_excluded in node_slices:
            pool, seen_norms = [], {normalize_query_for_dedup(q) for q in all_excluded}
            for comp in raw_completions[start_idx:end_idx]:
                for q in extract_valid_queries(comp, max_queries=k_queries):
                    norm = normalize_query_for_dedup(q)
                    if q not in pool and q not in all_excluded and norm not in seen_norms:
                        seen_norms.add(norm)
                        pool.append(q)
            item["candidates"] = pool

    def get_votes_matrix(self, texts: List[str], query: str, majority_votes: int = 1) -> Optional[np.ndarray]:
        q_map = self.record_vote_cache.get((majority_votes, normalize_query_for_dedup(query)))
        if not q_map or not all(t in q_map for t in texts):
            return None
        return np.array([q_map[t] for t in texts], dtype=int)

    def answer_queries_cached(
        self, full_texts: List[str], queries: List[str], chunk_size: Optional[int] = None, majority_votes: int = 1, vote_temperature: float = 0.2
    ) -> pd.DataFrame:
        if not queries:
            return pd.DataFrame(index=np.arange(len(full_texts)))
        chunk_size = chunk_size or getattr(self, "default_chunk_size", 8)
        sampling_temp = vote_temperature if majority_votes > 1 else 0.0

        unique_eval, seen_norms = [], set()
        for q in queries:
            norm_q = normalize_query_for_dedup(q)
            if norm_q not in seen_norms:
                seen_norms.add(norm_q)
                unique_eval.append((q, norm_q))

        needing_eval = [
            q for q, nq in unique_eval
            if not all(txt in self.record_vote_cache.get((majority_votes, nq), {}) for txt in full_texts)
        ]
        for start in range(0, len(needing_eval), chunk_size):
            q_chunk = needing_eval[start : start + chunk_size]
            c_norms = [normalize_query_for_dedup(q) for q in q_chunk]
            missing = [
                txt for txt in dict.fromkeys(full_texts)
                if any(txt not in self.record_vote_cache.get((majority_votes, nq), {}) for nq in c_norms)
            ]
            if not missing:
                continue
            k_c = len(c_norms)
            stop = ["\n\n", "Note:", "Explanation:", "Reasoning:", "Question:", "User:", "<|im_end|>", f"\n{k_c + 1}.", f"{k_c + 1}.", "Record", "RECORD", "EXAMPLE"]
            raw_out = self.generate_batch_text(
                [build_multi_answer_prompt(txt, q_chunk) for txt in missing],
                max_tokens=6 if k_c == 1 else (k_c * 5 + 2),
                temperature=sampling_temp,
                n=majority_votes,
                stop=stop,
            )
            for rec_txt, item_out in zip(missing, raw_out):
                comps = item_out if majority_votes > 1 else [item_out]
                vote_mat = np.array([extract_answers_from_llm_response(ans, k_c) for ans in comps], dtype=int)
                for col_idx, nq in enumerate(c_norms):
                    self.record_vote_cache.setdefault((majority_votes, nq), {})[rec_txt] = vote_mat[:, col_idx]

        cols = [(np.mean(self.get_votes_matrix(full_texts, q, majority_votes), axis=1) >= 0.5).astype(int) for q in queries]
        return pd.DataFrame(np.column_stack(cols), columns=[f"Q{i}" for i in range(len(queries))])
