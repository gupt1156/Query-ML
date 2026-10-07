
import re

# 1. Generator.py
with open('/scratch/rvg9413/Query-ML/src/generator.py', 'r') as f:
    text = f.read()

count1 = text.count('(Yes|No) ')
print('Found (Yes|No) occurrences in generator.py:', count1)
text = text.replace('r"(\\d+)[\\.\\)\\:\\-]?\\s*(Yes|No) "', 'r"(\\d+)[\\.\\)\\:\\-]?\\s*[\\*_]*(Yes|No)\\b"')
text = text.replace('r" (Yes|No) "', 'r"\\b(Yes|No)\\b"')

text = text.replace(
    '        vote_temperature: float = 0.5\n    ) -> pd.DataFrame:',
    '        vote_temperature: float = 0.5,\n        temperature: Optional[float] = None\n    ) -> pd.DataFrame:'
)

old_temp_block = '''            if majority_votes > 1:
                # Stochastic passes with majority voting
                sample_matrices = []
                for v_idx in range(majority_votes):
                    raw_answers = self.generate_batch_text(prompts, max_tokens=max_tokens, temperature=vote_temperature)
                    sample_mat = np.array([
                        extract_answers_from_llm_response(ans_text, len(query_chunk))
                        for ans_text in raw_answers
                    ], dtype=int)
                    sample_matrices.append(sample_mat)

                stacked = np.stack(sample_matrices, axis=0) # (majority_votes, n_records, len(chunk))
                answer_matrix = (np.mean(stacked, axis=0) >= 0.5).astype(int)
            else:
                raw_answers = self.generate_batch_text(prompts, max_tokens=max_tokens, temperature=0.0)'''

new_temp_block = '''            sampling_temp = temperature if temperature is not None else (vote_temperature if majority_votes > 1 else 0.0)

            if majority_votes > 1:
                # Stochastic passes with majority voting
                sample_matrices = []
                for v_idx in range(majority_votes):
                    raw_answers = self.generate_batch_text(prompts, max_tokens=max_tokens, temperature=sampling_temp)
                    sample_mat = np.array([
                        extract_answers_from_llm_response(ans_text, len(query_chunk))
                        for ans_text in raw_answers
                    ], dtype=int)
                    sample_matrices.append(sample_mat)

                stacked = np.stack(sample_matrices, axis=0) # (majority_votes, n_records, len(chunk))
                answer_matrix = (np.mean(stacked, axis=0) >= 0.5).astype(int)
            else:
                raw_answers = self.generate_batch_text(prompts, max_tokens=max_tokens, temperature=sampling_temp)'''

if old_temp_block in text:
    text = text.replace(old_temp_block, new_temp_block)
    print("Replaced temp block in generator.py")

with open('/scratch/rvg9413/Query-ML/src/generator.py', 'w') as f:
    f.write(text)

# 2. Reliability.py
with open('/scratch/rvg9413/Query-ML/src/reliability.py', 'r') as f:
    rtext = f.read()

old_rel = 's_df = engine.answer_queries_cached(\n            texts, valid_queries, cache_key_prefix=f"rel_stoch_{split_name}_run{r}"\n        )'
new_rel = 's_df = engine.answer_queries_cached(\n            texts, valid_queries, cache_key_prefix=f"rel_stoch_{split_name}_run{r}", temperature=temperature\n        )'
if old_rel in rtext:
    rtext = rtext.replace(old_rel, new_rel)
    print("Replaced stoch call in reliability.py")

with open('/scratch/rvg9413/Query-ML/src/reliability.py', 'w') as f:
    f.write(rtext)

# 3. Tree.py
with open('/scratch/rvg9413/Query-ML/src/tree.py', 'r') as f:
    ttext = f.read()

old_corr = 'if corr > max_correlation:'
new_corr = 'if not np.isnan(corr) and corr > max_correlation:'
if old_corr in ttext:
    ttext = ttext.replace(old_corr, new_corr)
    print("Guarded corr in tree.py")

with open('/scratch/rvg9413/Query-ML/src/tree.py', 'w') as f:
    f.write(ttext)

