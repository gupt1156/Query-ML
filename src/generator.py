"""
High-Performance LLM Generation and Answering Engine backed by vLLM.

Key Functionality:
1. Batched KV-Cached generation over contrastive dataset subsets.
2. Cached multi-answer query verification against candidate record pools.
3. Majority-vote self-consistency answering to suppress LLM flip noise.
4. Non-redundant candidate generation conditioned on branch ancestor history.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams

from src.prompts import build_multi_answer_prompt, build_query_generation_prompt


def sanitize_query_line(raw_line: str) -> Optional[str]:
    cleaned = raw_line.strip()
    if not cleaned:
        return None
    cleaned = re.sub(r'^\s*(?:\d+[\.\)\:-]|[\*\-\•])\s*', '', cleaned).strip()
    for q_char in ['"', "'", '`', '“', '”']:
        cleaned = cleaned.strip(q_char).strip()
    if "?" in cleaned:
        cleaned = cleaned.split("?")[0].strip() + "?"
    else:
        cleaned += "?"

    # Clean awkward phrasing that confused 7B model extraction
    cleaned = re.sub(r'\bat least (\d+) number of\b', r'at least \1', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b(\d+) number of\b', r'\1', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b1 publications\b', '1 publication', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b1 activities\b', '1 activity', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b1 letters\b', '1 letter', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    if not cleaned.lower().startswith(("does", "is", "has", "did", "are", "was", "were", "can")):
        return None

    if len(cleaned.split()) < 4 or len(cleaned) < 15:
        return None

    return cleaned


def normalize_query_for_dedup(query: str) -> str:
    """Normalizes query text to catch semantic near-duplicates and paraphrases."""
    q = query.lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    # Map common synonyms in admissions context
    q = q.replace("indicate", "show").replace("list", "show").replace("mention", "show")
    q = q.replace("a math grade", "math grade").replace("an a", "a")
    return q


def extract_valid_queries(raw_llm_output: str, max_queries: int = 5) -> List[str]:
    extracted: List[str] = []
    seen_normalized = set()

    for line in raw_llm_output.strip().split("\n"):
        sanitized = sanitize_query_line(line)
        if sanitized is not None:
            norm = normalize_query_for_dedup(sanitized)
            if norm not in seen_normalized:
                seen_normalized.add(norm)
                extracted.append(sanitized)
                if len(extracted) >= max_queries:
                    break
    return extracted


def parse_binary_answer_line(line: str, expected_count: int) -> Optional[Tuple[int, int]]:
    stripped_line = line.strip()
    if not stripped_line:
        return None

    match = re.search(r"(\d+)[\.\)\:\-]?\s*[\*_]*(Yes|No)\b", stripped_line, re.IGNORECASE)
    if match:
        index_num = int(match.group(1)) - 1
        binary_value = 1 if match.group(2).lower() == "yes" else 0
        if 0 <= index_num < expected_count:
            return index_num, binary_value

    binary_match = re.search(r"\b(Yes|No)\b", stripped_line, re.IGNORECASE)
    if binary_match:
        binary_value = 1 if binary_match.group(1).lower() == "yes" else 0
        return 0, binary_value

    return None


def extract_answers_from_llm_response(raw_llm_output: str, num_queries: int) -> List[int]:
    answers = [0] * num_queries
    for line in raw_llm_output.strip().split("\n"):
        parsed = parse_binary_answer_line(line, num_queries)
        if parsed is not None:
            question_idx, binary_val = parsed
            answers[question_idx] = binary_val
    return answers


def sample_contrastive_record_pairs(
    df: pd.DataFrame,
    num_subsets: int,
    samples_per_group: int = 2,
    base_seed: int = 42
) -> List[Tuple[List[str], List[str]]]:
    positive_df = df[df["target"] == 1]
    negative_df = df[df["target"] == 0]

    if len(positive_df) == 0 or len(negative_df) == 0:
        return []

    sampled_pairs: List[Tuple[List[str], List[str]]] = []
    n_pos = min(samples_per_group, len(positive_df))
    n_neg = min(samples_per_group, len(negative_df))

    for subset_idx in range(num_subsets):
        subset_seed = base_seed + 10 * subset_idx
        pos_samples = positive_df.sample(n=n_pos, random_state=subset_seed)["text"].tolist()
        neg_samples = negative_df.sample(n=n_neg, random_state=subset_seed)["text"].tolist()
        sampled_pairs.append((pos_samples, neg_samples))

    return sampled_pairs


class FastVLLMEngine:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        gpu_memory_utilization: float = 0.85
    ):
        local_cluster_snapshot = (
            "/scratch/rvg9413/.cache/huggingface/hub/"
            "models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
            "a09a35458c702b33eeacc393d103063234e8bc28"
        )
        resolved_model_path = local_cluster_snapshot if os.path.exists(local_cluster_snapshot) else model_name

        print(f"Initializing FastVLLMEngine with model: {resolved_model_path}")
        self.llm = LLM(
            model=resolved_model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=1024,
            trust_remote_code=True,
            enforce_eager=True
        )
        self.query_cache: Dict[str, np.ndarray] = {}
        self.raw_votes_cache: Dict[str, np.ndarray] = {}

    def generate_batch_text(
        self,
        prompts: List[str],
        max_tokens: int = 160,
        temperature: float = 0.3,
        n: int = 1,
        stop: Optional[List[str]] = None
    ) -> Any:
        stop_tokens = stop if stop is not None else ["\n\n\n", "User:", "<|im_end|>"]
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            stop=stop_tokens
        )
        generation_outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        if n == 1:
            return [out.outputs[0].text.strip() for out in generation_outputs]
        return [[cand.text.strip() for cand in out.outputs] for out in generation_outputs]

    def generate_queries(
        self,
        df: pd.DataFrame,
        n_subsets: int = 8,
        k_queries: int = 5,
        seed: int = 42,
        exclude: Optional[List[str]] = None,
        ancestor_queries: Optional[List[str]] = None
    ) -> List[str]:
        record_pairs = sample_contrastive_record_pairs(
            df=df,
            num_subsets=n_subsets,
            samples_per_group=2,
            base_seed=seed
        )
        if not record_pairs:
            return []

        all_excluded = set(exclude or [])
        if ancestor_queries:
            all_excluded.update(ancestor_queries)

        prompts = [
            build_query_generation_prompt(
                pos_texts, 
                neg_texts, 
                n_queries=k_queries,
                ancestor_queries=ancestor_queries
            )
            for pos_texts, neg_texts in record_pairs
        ]

        raw_completions = self.generate_batch_text(prompts, max_tokens=180, temperature=0.3)

        candidate_pool: List[str] = []
        seen_normalized = {normalize_query_for_dedup(q) for q in all_excluded}

        for completion in raw_completions:
            parsed_queries = extract_valid_queries(completion, max_queries=k_queries)
            for query in parsed_queries:
                norm = normalize_query_for_dedup(query)
                if query not in candidate_pool and query not in all_excluded and norm not in seen_normalized:
                    seen_normalized.add(norm)
                    candidate_pool.append(query)

        return candidate_pool

    def answer_queries_cached(
        self,
        full_texts: List[str],
        queries: List[str],
        active_indices: Optional[np.ndarray] = None,
        cache_key_prefix: str = "train",
        chunk_size: int = 16,
        majority_votes: int = 1,
        vote_temperature: float = 0.2,
        temperature: Optional[float] = None
    ) -> pd.DataFrame:
        if active_indices is None:
            active_indices = np.arange(len(full_texts))

        if not queries:
            return pd.DataFrame(index=active_indices)

        prefix = f"{cache_key_prefix}_maj{majority_votes}" if majority_votes > 1 else cache_key_prefix

        uncached_queries = [
            q for q in queries
            if f"{prefix}::{q}" not in self.query_cache
        ]

        for chunk_start in range(0, len(uncached_queries), chunk_size):
            query_chunk = uncached_queries[chunk_start : chunk_start + chunk_size]
            prompts = [
                build_multi_answer_prompt(record_text, query_chunk)
                for record_text in full_texts
            ]
            max_tokens = max(24, len(query_chunk) * 6)

            sampling_temp = temperature if temperature is not None else (vote_temperature if majority_votes > 1 else 0.0)
            if majority_votes > 1:
                # Native parallel stochastic passes in vLLM with majority voting
                answer_stop = ["\n\n", "Note:", "Explanation:", "Reasoning:", "Question:", "User:", "<|im_end|>"]
                raw_multi = self.generate_batch_text(
                    prompts, max_tokens=max_tokens, temperature=sampling_temp, n=majority_votes, stop=answer_stop
                )
                answer_rows = []
                all_votes_list = []
                for rec_completions in raw_multi:
                    vote_mat = np.array([
                        extract_answers_from_llm_response(ans_text, len(query_chunk))
                        for ans_text in rec_completions
                    ], dtype=int)
                    maj_row = (np.mean(vote_mat, axis=0) >= 0.5).astype(int)
                    answer_rows.append(maj_row)
                    all_votes_list.append(vote_mat.T)
                answer_matrix = np.array(answer_rows, dtype=int)
                all_votes_tensor = np.array(all_votes_list, dtype=int)
            else:
                answer_stop = ["\n\n", "Note:", "Explanation:", "Reasoning:", "Question:", "User:", "<|im_end|>"]
                raw_answers = self.generate_batch_text(prompts, max_tokens=max_tokens, temperature=sampling_temp, stop=answer_stop)
                answer_matrix = np.array([
                    extract_answers_from_llm_response(ans_text, len(query_chunk))
                    for ans_text in raw_answers
                ], dtype=int)
                all_votes_tensor = answer_matrix[:, :, None]

            for col_idx, query_str in enumerate(query_chunk):
                cache_key = f"{prefix}::{query_str}"
                self.query_cache[cache_key] = answer_matrix[:, col_idx]
                self.raw_votes_cache[cache_key] = all_votes_tensor[:, col_idx, :]

        selected_columns = [
            self.query_cache[f"{prefix}::{q}"][active_indices]
            for q in queries
        ]
        result_matrix = np.column_stack(selected_columns)
        column_labels = [f"Q{i}" for i in range(len(queries))]

        return pd.DataFrame(result_matrix, columns=column_labels, index=active_indices)





