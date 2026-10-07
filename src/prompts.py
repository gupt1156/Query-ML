"""
Domain-Agnostic Prompt Templates for Statistical Query Generation and Verification.

This module provides inductive synthesis templates:
1. Contrastive Query Generation Prompt: Instructs the LLM to inspect contrastive positive
   (Group A) and negative (Group B) records and formulate atomic, factual yes/no questions.
2. Multi-Answer Verification Prompt: Instructs the LLM to evaluate a batch of yes/no questions
   against an individual record with strict field isolation and zero hallucination.
"""

from typing import List, Optional


def format_contrastive_record_group(record_texts: List[str], group_label: str) -> str:
    formatted_blocks = [
        f"Record {group_label}{idx + 1}:\n{text.strip()}"
        for idx, text in enumerate(record_texts)
    ]
    return "\n---\n".join(formatted_blocks)


def format_numbered_query_list(queries: List[str]) -> str:
    return "\n".join([f"{idx + 1}. {q.strip()}" for idx, q in enumerate(queries)])


def build_query_generation_prompt(
    pos_texts: List[str],
    neg_texts: List[str],
    n_queries: int = 5,
    ancestor_queries: Optional[List[str]] = None
) -> str:
    group_a_block = format_contrastive_record_group(pos_texts, "A")
    group_b_block = format_contrastive_record_group(neg_texts, "B")

    ancestor_constraint = ""
    if ancestor_queries and len(ancestor_queries) > 0:
        cleaned_anc = [f"- {q.strip()}" for q in ancestor_queries if q.strip()]
        ancestor_list_str = "\n".join(cleaned_anc)
        ancestor_constraint = f"""
---
ALREADY EVALUATED QUESTIONS IN THIS BRANCH:
{ancestor_list_str}

MANDATORY NON-REDUNDANCY RULE:
- Do NOT generate questions testing any of the attributes or conditions evaluated in the questions above.
- Do NOT generate paraphrases, variations, or alternative thresholds of the questions above.
- Focus strictly on NOVEL, UNTESTED fields (e.g. testing different academic or extracurricular attributes).
"""

    return f"""You are an expert statistical machine learning researcher.

Your task is to identify objective, verifiable yes/no questions that distinguish Group A (Positive Records) from Group B (Negative Records) based strictly on the text provided.

RULES:
1. STRICT FACTUAL PRESENCE: Only ask about concrete facts, explicit keywords, specific values, or conditions directly stated in the text.
2. NO SUBJECTIVE INFERENCE: Do NOT ask subjective or predictive questions.
3. CONTRASTIVE SEPARATION: Each question must produce OPPOSITE answers between Group A and Group B. You may test an attribute present in Group A, OR an attribute present in Group B but absent in Group A. Both directions yield maximal separation.
4. DIVERSITY: Each question must focus on a distinct attribute or field. Do NOT create duplicate or redundant questions.
5. ATOMIC ATTRIBUTES ONLY: Each question must evaluate strictly ONE individual attribute or field. Do NOT combine multiple attributes using "and", "or", or compound clauses.
6. THRESHOLDS & DIRECT PHRASING: For categorical fields, test the specific value that distinguishes the groups (e.g., 'Does the record indicate [Attribute] is [Value]?'). For numerical quantities, prefer comparative phrasing (e.g., 'Does the record show at least X [Metric]?' or 'Does the record list X or more [Item]?') rather than phrasing such as 'Is the number of...'. Do not include literal brackets or mention Group A or Group B in the questions.
7. MINIMAL SEPARATION THRESHOLDS: For numerical attributes where one group has positive values and the other group has zero or lower values, always test the lowest threshold boundary that cleanly separates the groups (e.g., test 'at least 1' or '1 or more' when distinguishing positive values from zero; 'at least 2' when distinguishing values >= 2 from values <= 1) rather than higher arbitrary values.
{ancestor_constraint}
---
GROUP A (Positive Records):
{group_a_block}

---
GROUP B (Negative Records):
{group_b_block}

---
Generate exactly {n_queries} distinct, highly discriminative yes/no questions.
Output numbered questions from 1 to {n_queries}, each on a separate line ending with a question mark.
"""


def build_multi_answer_prompt(record_text: str, queries: List[str]) -> str:
    numbered_queries = format_numbered_query_list(queries)

    return f"""Evaluate each yes/no question based strictly on the student record provided below.

RULES:
1. STRICT FORMAT: Output ONLY the question number followed by 'Yes' or 'No' (e.g. '1. Yes').
2. NO COMMENTARY: Do NOT output any explanation, notes, or additional words.
3. ZERO RULE: If the record states a value is 0, questions asking 'at least 1' or 'present' must be answered 'No'.
4. THRESHOLD RULE: 'at least X' or 'X or more' means value >= X.

EXAMPLE:
Student record:
Math grade: A
Number of publications: 0
Number of extracurricular activities: 2
Questions:
1. Does the record show a Math grade of A?
2. Does the record show at least 1 publication?
3. Does the record show at least 2 extracurricular activities?
Output:
1. Yes
2. No
3. Yes

STUDENT RECORD:
{record_text.strip()}

QUESTIONS:
{numbered_queries}

OUTPUT:"""

