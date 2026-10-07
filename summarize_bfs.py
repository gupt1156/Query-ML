
import glob
import json
import numpy as np

paths = sorted(glob.glob('/scratch/rvg9413/Query-ML/results/vllm_regimes_maj3_local_bfs_contextual_seed*/full_experiment_data.json'))
print(f'Total seeds completed: {len(paths)} / 10\n')

header = f"{'Seed':<6} | {'Local Acc':<10} | {'Local AUC':<10} | {'Splits':<6} | {'Causal':<6} | {'Distr':<6} | {'CausalRec':<10} | {'Time (s)':<10}"
print(header)
print('-' * len(header))

local_accs, local_aucs, causal_recs, times = [], [], [], []

for p in paths:
    seed = p.split('_seed')[-1].split('/')[0]
    with open(p) as f:
        d = json.load(f)
    lr = d['local_runs'][0]
    timing = d.get('timing', {}).get('local_seconds', 0.0)
    
    l_acc = lr['accuracy']
    l_auc = lr['roc_auc']
    splits = lr['total_splits']
    causal = lr['causal_splits']
    distr = lr['distractor_splits']
    crec = lr['causal_recovery_rate']

    local_accs.append(l_acc)
    local_aucs.append(l_auc)
    causal_recs.append(crec)
    times.append(timing)

    print(f"{seed:<6} | {l_acc:<10.3f} | {l_auc:<10.3f} | {splits:<6} | {causal:<6} | {distr:<6} | {crec*100:<9.1f}% | {timing:<10.1f}")

if local_accs:
    print('-' * len(header))
    print(f"{'MEAN':<6} | {np.mean(local_accs):<10.3f} | {np.mean(local_aucs):<10.3f} |        |        |        | {np.mean(causal_recs)*100:<9.1f}% | {np.mean(times):<10.1f}")
    print(f"{'STD':<6} | {np.std(local_accs):<10.3f} | {np.std(local_aucs):<10.3f} |        |        |        | {np.std(causal_recs)*100:<9.1f}% | {np.std(times):<10.1f}")

