# -*- coding: utf-8 -*-
"""
1349轮标注完整分析 + 可视化
布局: 3x2 = 6个子图
"""
import json, re, os, sys
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
LOG_PATH = os.path.join(ROOT_DIR, 'data', 'label_log.jsonl')
LABEL_PATH = os.path.join(ROOT_DIR, 'data', 'labeled.txt')
ANSWER_PATH = os.path.join(ROOT_DIR, 'data', 'answers.txt')
CHART_DIR = os.path.join(ROOT_DIR, 'charts')

# Font
for f in ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf']:
    if os.path.exists(f):
        plt.rcParams['font.family'] = fm.FontProperties(fname=f).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False

# ===== Load data =====
with open(LOG_PATH, 'r', encoding='utf-8') as f:
    entries = [json.loads(l) for l in f]

with open(LABEL_PATH, 'r', encoding='utf-8') as f:
    labels = [l.strip() for l in f if l.strip()]

# Ground truth
answers = []
with open(ANSWER_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        m = re.search(r'【([^】]+)】', line)
        if m:
            answers.append(m.group(1).strip())

# ===== Compare =====
N = min(len(labels), len(answers))
correct_flags = []
wrong_indices = []
for i in range(N):
    label_parts = labels[i].split("|")
    acceptable = answers[i].split("|")
    if any(part.strip() in acceptable for part in label_parts):
        correct_flags.append(1)
    else:
        correct_flags.append(0)
        wrong_indices.append(i)

accuracy = sum(correct_flags) / N * 100

# ===== Per-section accuracy (every 50 rounds) =====
bin_size = 50
num_bins = (N + bin_size - 1) // bin_size
bin_acc = []
bin_centers = []
for b in range(num_bins):
    start = b * bin_size
    end = min(start + bin_size, N)
    chunk = correct_flags[start:end]
    acc = sum(chunk) / len(chunk) * 100 if chunk else 0
    bin_acc.append(acc)
    bin_centers.append((start + end) // 2)

# ===== Extract timeline data =====
seq_tool = []
seq_tok = []
seq_lines = []
seq_speaker = []
lines_tool = []
lines_tok = []
for idx, e in enumerate(entries):
    ln = e.get('dialogue_line', 0)
    tc = len(e.get('tool_calls', []))
    tok = e.get('agents', {}).get('Labeler', {}).get('total_tokens', 0)
    sp = e.get('result', {}).get('speaker', '?')
    seq_tool.append(tc)
    seq_tok.append(tok)
    seq_lines.append(ln)
    seq_speaker.append(sp)
    lines_tool.append(tc)
    lines_tok.append(tok)

# Token bins (every 50)
tok_bins = []
for b in range(num_bins):
    start = b * bin_size
    end = min(start + bin_size, N)
    chunk = seq_tok[start:end]
    tok_bins.append(np.mean(chunk) if chunk else 0)

# Tool percentage bins
tool_bins = []
for b in range(num_bins):
    start = b * bin_size
    end = min(start + bin_size, N)
    chunk = seq_tool[start:end]
    pct = sum(chunk) / len(chunk) * 100 if chunk else 0
    tool_bins.append(pct)

# Speaker distribution (all labels)
spk_counter = Counter()
for e in entries:
    sp = e.get('result', {}).get('speaker', '?')
    if sp and sp != '?':
        spk_counter[sp] += 1

# Error confusion
wrong_label_counter = Counter()
for idx in wrong_indices:
    wrong_label_counter[labels[idx]] += 1
wrong_answer_counter = Counter()
for idx in wrong_indices:
    wrong_answer_counter[answers[idx]] += 1

# ===== Context utilization =====
ctx_limit = 40960
ctx_usage = [min(t / ctx_limit * 100, 100) for t in seq_tok]

# ===== Binned context usage =====
ctx_bins = []
for b in range(num_bins):
    start = b * bin_size
    end = min(start + bin_size, N)
    chunk = ctx_usage[start:end]
    ctx_bins.append(np.mean(chunk) if chunk else 0)

# ===== Plot: 3x2 =====
fig, axes = plt.subplots(3, 2, figsize=(18, 15))
fig.suptitle('NovelSpeakerV4 v2.1 — Complete 1349-Round Analysis', fontsize=16, fontweight='bold', y=0.98)

colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6ABD5B', '#4A6FA5']
C_ACC = '#2E86AB'
C_TOOL = '#A23B72'
C_TOK = '#F18F01'
C_CTX = '#C73E1D'
C_ERR = '#D9534F'

bin_labels = []
for b in range(num_bins):
    start = b * bin_size + 1
    end = min((b + 1) * bin_size, N)
    bin_labels.append(f'{start}\n~{end}')

# (1,1) Accuracy trend per 50 rounds
ax1 = axes[0, 0]
bars = ax1.bar(range(num_bins), bin_acc, color=C_ACC, width=0.6, edgecolor='white', linewidth=0.3)
ax1.axhline(y=accuracy, color='gray', linestyle='--', linewidth=1, alpha=0.6)
ax1.text(num_bins - 1, accuracy + 0.8, f'Overall: {accuracy:.1f}%', fontsize=9, color='gray', ha='right')
ax1.set_xticks(range(num_bins))
ax1.set_xticklabels(bin_labels, fontsize=6, rotation=30)
ax1.set_ylabel('Accuracy (%)', fontsize=11)
ax1.set_ylim(0, 105)
ax1.set_title('Accuracy per 50 Rounds', fontsize=12, fontweight='bold')
for bar, val in zip(bars, bin_acc):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.0f}%', ha='center', va='bottom', fontsize=6, fontweight='bold')

# (1,2) Tool call rate per 50 rounds
ax2 = axes[0, 1]
bars = ax2.bar(range(num_bins), tool_bins, color=C_TOOL, width=0.6, edgecolor='white', linewidth=0.3)
ax2.set_xticks(range(num_bins))
ax2.set_xticklabels(bin_labels, fontsize=6, rotation=30)
ax2.set_ylabel('Rounds with Tool Calls (%)', fontsize=11)
ax2.set_ylim(0, 105)
ax2.set_title('Tool Call Rate per 50 Rounds', fontsize=12, fontweight='bold')
overall_tool = sum(seq_tool) / len(seq_tool) * 100
ax2.axhline(y=overall_tool, color='gray', linestyle='--', linewidth=1, alpha=0.6)
ax2.text(num_bins - 1, overall_tool + 1.5, f'Overall: {overall_tool:.0f}%', fontsize=9, color='gray', ha='right')

# (2,1) Token consumption trend (scatter + moving avg)
ax3 = axes[1, 0]
window = 20
weights = np.ones(window) / window
if len(seq_tok) >= window:
    smooth = np.convolve(seq_tok, weights, mode='valid')
    smooth_x = np.arange(window // 2, window // 2 + len(smooth))
    ax3.plot(range(len(seq_tok)), seq_tok, alpha=0.15, color=C_TOK, linewidth=0.5)
    ax3.plot(smooth_x, smooth, color=C_TOK, linewidth=2, label=f'{window}-round moving avg')
else:
    ax3.plot(range(len(seq_tok)), seq_tok, color=C_TOK, linewidth=1)
ax3.axhline(y=ctx_limit, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax3.text(len(seq_tok) - 5, ctx_limit + 300, '40K context limit', fontsize=8, color='red', ha='right')
ax3.set_xlabel('Round', fontsize=11)
ax3.set_ylabel('Tokens', fontsize=11)
ax3.set_title('Token Consumption per Round', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9, loc='upper left')
ax3.text(0.98, 0.08, f'Avg: {np.mean(seq_tok):.0f}\nMax: {max(seq_tok):,}',
         transform=ax3.transAxes, ha='right', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# (2,2) Context window utilization
ax4 = axes[1, 1]
ax4.fill_between(range(len(ctx_usage)), ctx_usage, alpha=0.2, color=C_CTX)
ax4.plot(range(len(ctx_usage)), ctx_usage, color=C_CTX, linewidth=0.8, alpha=0.6)
window2 = 30
if len(ctx_usage) >= window2:
    smooth_ctx = np.convolve(ctx_usage, np.ones(window2)/window2, mode='valid')
    smooth_ctx_x = np.arange(window2//2, window2//2 + len(smooth_ctx))
    ax4.plot(smooth_ctx_x, smooth_ctx, color=C_CTX, linewidth=2, label=f'{window2}-round avg')
ax4.axhline(y=62, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax4.text(len(ctx_usage) - 5, 63, 'Peak: 62%', fontsize=8, color='red', ha='right')
ax4.set_xlabel('Round', fontsize=11)
ax4.set_ylabel('Context Window Usage (%)', fontsize=11)
ax4.set_title('Context Window Utilization (40K)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9, loc='upper left')
ax4.text(0.98, 0.08, f'Avg: {np.mean(ctx_usage):.1f}%', transform=ax4.transAxes,
         ha='right', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# (3,1) Top speakers distribution
ax5 = axes[2, 0]
top_n = 10
top_speakers = spk_counter.most_common(top_n)
spk_names = [s[0] for s in top_speakers]
spk_counts = [s[1] for s in top_speakers]
colors_spk = plt.cm.Paired(np.linspace(0, 1, len(spk_names)))
bars = ax5.barh(range(len(spk_names)), spk_counts, color=colors_spk, edgecolor='white', linewidth=0.5)
ax5.set_yticks(range(len(spk_names)))
ax5.set_yticklabels(spk_names, fontsize=9)
ax5.set_xlabel('Occurrences', fontsize=11)
ax5.set_title('Top 10 Speakers', fontsize=12, fontweight='bold')
for bar, val in zip(bars, spk_counts):
    ax5.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
             str(val), ha='left', va='center', fontsize=8)
# Show total unique
total_unique = len(spk_counter)
ax5.text(0.98, 0.08, f'Unique: {total_unique}', transform=ax5.transAxes,
         ha='right', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# (3,2) Error distribution
ax6 = axes[2, 1]
if wrong_indices:
    # Top 10 most confused ground truths
    top_errors = wrong_answer_counter.most_common(10)
    err_names = [f'GT: {s[0][:12]}' for s in top_errors]
    err_counts = [s[1] for s in top_errors]
    colors_err = plt.cm.Reds(np.linspace(0.3, 0.8, len(err_names)))
    bars = ax6.barh(range(len(err_names)), err_counts, color=colors_err, edgecolor='white', linewidth=0.5)
    ax6.set_yticks(range(len(err_names)))
    ax6.set_yticklabels(err_names, fontsize=8)
    ax6.set_xlabel('Error Count', fontsize=11)
    ax6.set_title(f'Top Confused Labels ({len(wrong_indices)} total errors)', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, err_counts):
        ax6.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 str(val), ha='left', va='center', fontsize=8)
else:
    ax6.text(0.5, 0.5, 'No errors found!', ha='center', va='center', fontsize=14, color='green')
    ax6.set_title('Error Analysis', fontsize=12, fontweight='bold')

# Summary
summary = (
    f"Summary: {N} rounds validated | Accuracy: {accuracy:.1f}% | "
    f"Errors: {len(wrong_indices)} | Tool call rate: {overall_tool:.1f}% | "
    f"Avg token: {np.mean(seq_tok):.0f} | Unique speakers: {total_unique}"
)
fig.text(0.5, 0.005, summary, ha='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray'))

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
out_path = os.path.join(CHART_DIR, 'full_analysis.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Chart saved: {out_path}')
print(f'Accuracy: {accuracy:.1f}% ({len(wrong_indices)} errors out of {N})')
print(f'Tool call rate: {sum(seq_tool)}/{len(seq_tool)} ({overall_tool:.1f}%)')
print(f'Avg tokens: {np.mean(seq_tok):.0f}, Max: {max(seq_tok):,}')
plt.close()