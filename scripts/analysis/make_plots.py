
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2

# ----------------------------------------------------
# Plot 1: Reliability Breakdown (Reading vs Consistency)
# ----------------------------------------------------
categories = [
    'Publications\n(Causal)',
    'Extracurriculars\n(Causal)',
    'Math Grade\n(Causal)',
    'Rec Letters\n(Distractor)',
    'Service Hours\n(Distractor)'
]
reading_acc = [93.5, 89.9, 89.6, 94.5, 68.9]
consistency = [99.6, 99.6, 99.9, 99.8, 99.1]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10.5, 4.2), dpi=300)
rects1 = ax.bar(x - width/2, reading_acc, width, label='Reading Accuracy vs. True Data',
                color='#2b6cb0', edgecolor='#1a365d', linewidth=1.2, alpha=0.9)
rects2 = ax.bar(x + width/2, consistency, width, label='Consistency Across Runs (T=0.2)',
                color='#2f855a', edgecolor='#1c4532', linewidth=1.2, alpha=0.9)

ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold', labelpad=8)
ax.set_title('LLM Extraction Reliability Across Verified Tree Splits (Held-Out Test Set, N=200)',
             fontsize=12.5, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10.5, fontweight='bold')
ax.set_ylim(0, 112)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
ax.set_axisbelow(True)
ax.legend(loc='lower left', fontsize=10.5, framealpha=0.95, edgecolor='#cccccc')

for rect, val in zip(rects1, reading_acc):
    h = rect.get_height()
    ax.annotate(f'{val:.1f}%',
                xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1a365d')

for rect, val in zip(rects2, consistency):
    h = rect.get_height()
    ax.annotate(f'{val:.1f}%',
                xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1c4532')

plt.tight_layout()
fig.savefig('/scratch/rvg9413/Query-ML/results/fig_reliability_clean.png', bbox_inches='tight')
plt.close(fig)
print("Saved fig_reliability_clean.png")

# ----------------------------------------------------
# Plot 2: Benchmarks Across Regimes
# ----------------------------------------------------
regimes = ['Global CART\n(Baseline)', 'Local Greedy\n(No Voting)', 'Local 3-Vote\n(Fast)', 'Local 5-Vote\n(Our Model)']
mean_acc = [76.4, 78.2, 65.2, 83.7]
sd_acc = [3.0, 5.4, 12.0, 8.6]
peak_acc = [79.5, 84.5, 87.5, 97.0]

causal_recov = [73.3, 86.7, 90.0, 96.7]
causal_purity = [68.2, 72.5, 68.5, 80.0]

x = np.arange(len(regimes))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4), dpi=300)

# Subplot A
r1 = ax1.bar(x - width/2, mean_acc, width, yerr=sd_acc, capsize=4,
             label='10-Seed Mean Accuracy', color='#3182ce', edgecolor='#1a365d', linewidth=1.2, alpha=0.9)
r2 = ax1.bar(x + width/2, peak_acc, width,
             label='Peak Seed Accuracy', color='#38a169', edgecolor='#1c4532', linewidth=1.2, alpha=0.9)

ax1.set_ylabel('Test Accuracy (%)', fontsize=11.5, fontweight='bold', labelpad=8)
ax1.set_title('A. Test Accuracy Across 10 Seeds (Mean ± SD vs. Peak)', fontsize=11.5, fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(regimes, fontsize=9.5, fontweight='bold')
ax1.set_ylim(0, 112)
ax1.yaxis.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
ax1.set_axisbelow(True)
ax1.legend(loc='lower left', fontsize=9.5, framealpha=0.95, edgecolor='#cccccc')

for rect, val in zip(r1, mean_acc):
    h = rect.get_height()
    ax1.annotate(f'{val:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 12), textcoords="offset points", ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color='#1a365d')

for rect, val in zip(r2, peak_acc):
    h = rect.get_height()
    ax1.annotate(f'{val:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color='#1c4532')

# Subplot B
r3 = ax2.bar(x - width/2, causal_recov, width,
             label='Causal Rule Recovery (%)', color='#5a67d8', edgecolor='#2c3e50', linewidth=1.2, alpha=0.9)
r4 = ax2.bar(x + width/2, causal_purity, width,
             label='Causal Split Ratio (Purity %)', color='#dd6b20', edgecolor='#7b341e', linewidth=1.2, alpha=0.9)

ax2.set_ylabel('Percentage (%)', fontsize=11.5, fontweight='bold', labelpad=8)
ax2.set_title('B. Causal Rule Discovery & Branch Purity', fontsize=11.5, fontweight='bold', pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(regimes, fontsize=9.5, fontweight='bold')
ax2.set_ylim(0, 112)
ax2.yaxis.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
ax2.set_axisbelow(True)
ax2.legend(loc='lower left', fontsize=9.5, framealpha=0.95, edgecolor='#cccccc')

for rect, val in zip(r3, causal_recov):
    h = rect.get_height()
    ax2.annotate(f'{val:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color='#2c3e50')

for rect, val in zip(r4, causal_purity):
    h = rect.get_height()
    ax2.annotate(f'{val:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color='#7b341e')

plt.tight_layout()
fig.savefig('/scratch/rvg9413/Query-ML/results/fig_benchmarks_clean.png', bbox_inches='tight')
plt.close(fig)
print("Saved fig_benchmarks_clean.png")
