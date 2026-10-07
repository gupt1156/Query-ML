
import glob
import json
import numpy as np

paths = sorted(glob.glob('/scratch/rvg9413/Query-ML/results/vllm_regimes_maj5_contextual_seed*/full_experiment_data.json'))
print(f'Total seeds completed: {len(paths)} / 10\n')
header = f"{'Seed':<6} | {'Local Acc':<10} | {'Local AUC':<10} | {'Splits':<6} | {'Causal':<6} | {'Distr':<6} | {'CausalRec':<10} | {'Global Acc':<10}"
print(header)
print('-' * len(header))

local_accs, local_aucs, global_accs, causal_recs = [], [], [], []

for p in paths:
    seed = p.split('_seed')[-1].split('/')[0]
    with open(p) as f:
        d = json.load(f)
    lr = d['local_runs'][0]
    gr = d['global_runs'][0]
    
    l_acc = lr['accuracy']
    l_auc = lr['roc_auc']
    splits = lr['total_splits']
    causal = lr['causal_splits']
    distr = lr['distractor_splits']
    crec = lr['causal_recovery_rate']
    g_acc = gr['accuracy']

    local_accs.append(l_acc)
    local_aucs.append(l_auc)
    global_accs.append(g_acc)
    causal_recs.append(crec)

    print(f"{seed:<6} | {l_acc:<10.3f} | {l_auc:<10.3f} | {splits:<6} | {causal:<6} | {distr:<6} | {crec*100:<9.1f}% | {g_acc:<10.3f}")

if local_accs:
    print('-' * len(header))
    print(f"{'MEAN':<6} | {np.mean(local_accs):<10.3f} | {np.mean(local_aucs):<10.3f} |        |        |        | {np.mean(causal_recs)*100:<9.1f}% | {np.mean(global_accs):<10.3f}")

