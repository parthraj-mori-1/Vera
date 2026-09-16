"""
VERA — Presentation Visualizations
==================================
Generates high-resolution, presentation-ready figures from the VERA
UCF-Crime results (segment scores + vision features + ground truth).

Run AFTER you have:
  - Data/segment_level_score/*.json   (VLM segment scores)
  - Data/vision_features/*.npy        (ImageBind features)
  - Data/UCF_Eval.json                (ground truth)

Output: PNG files in ./figures/  (300 DPI, ready to screenshot / embed in PPT)
"""

import json
import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics import auc, roc_curve, precision_recall_curve

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
ann_root      = 'Data/UCF_Eval.json'
vision_folder = 'Data/vision_features/'
score_folder  = 'Data/segment_level_score/'
FIG_DIR       = 'figures'
FPS           = 30
THRESHOLD     = 0.5

os.makedirs(FIG_DIR, exist_ok=True)

# A clean, presentation-friendly style
plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'figure.autolayout': True,
})

BLUE   = '#2563eb'
RED    = '#dc2626'
GREEN  = '#16a34a'
GREY   = '#9ca3af'


# --------------------------------------------------------------------------- #
# Reused VERA post-processing helpers
# --------------------------------------------------------------------------- #
def gaussian_kernel_original(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaussian_smoothing(data, sigma):
    n = len(data)
    x = np.arange(n)
    return gaussian_kernel_original(x, int(n / 2), sigma) * data


def gaussian_kernel(size, sigma):
    k = np.exp(-np.linspace(-size // 2, size // 2, size) ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def gaussian_smooth_1d(data, size=5, sigma=1.0):
    return np.convolve(data, gaussian_kernel(size, sigma), mode='same')


def softmax(v):
    e = np.exp(v - np.max(v))
    return e / np.sum(e)


def build_gt(sample):
    gt = np.zeros(sample['length'])
    tl = sample['temporal_label']
    for i in range(0, len(tl), 2):
        if tl[i] != -1:
            s = tl[i]
            e = min(tl[i + 1], sample['length'])
            gt[s:e] = 1.0
    return gt


def process_video(sample):
    """Return (frame_scores, ground_truth) for one video."""
    key = sample['video'].split('/')[-1].split('.')[0]
    with open(score_folder + key + '.json') as f:
        score = json.load(f)

    v_array = np.load(vision_folder + key + '.npy')
    v_norms = np.linalg.norm(v_array, axis=1, keepdims=True)
    normalized = v_array / v_norms
    sim = np.dot(normalized, normalized.T)

    top_n = max(1, int(0.15 * sim.shape[0]))
    idx = np.argsort(sim, axis=1)[:, -top_n:]
    top_vals = np.take_along_axis(sim, idx, axis=1)

    keys = list(score.keys())
    raw = [score[k]['score'] for k in keys]

    refined = []
    for i in range(len(keys)):
        neigh = np.array([raw[n] for n in idx[i].tolist()])
        refined.append(np.dot(softmax(top_vals[i] * 10), neigh))
    refined = gaussian_smooth_1d(np.array(refined), 15, 10)

    frame_scores = []
    sr = 16
    for i, k in enumerate(keys):
        s = round(refined[i], 1)
        if i != len(keys) - 1:
            frame_scores.extend([s] * sr)
        else:
            frame_scores.extend([s] * int(score[k]['end'] - int(k)))
    frame_scores = np.array(frame_scores)
    frame_scores = gaussian_smoothing(frame_scores, int(len(frame_scores) / 2))

    gt = build_gt(sample)
    n = min(len(frame_scores), len(gt))
    return frame_scores[:n], gt[:n], key


# --------------------------------------------------------------------------- #
# Run over the whole test set
# --------------------------------------------------------------------------- #
with open(ann_root, encoding='utf-8') as f:
    annotation = json.load(f)

all_scores, all_gt = [], []
per_video = []            # (key, frame_scores, gt)

print('Processing videos...')
for sample in annotation:
    try:
        fs, gt, key = process_video(sample)
    except FileNotFoundError:
        continue
    all_scores.extend(fs)
    all_gt.extend(gt)
    per_video.append((key, fs, gt))

all_scores = np.array(all_scores)
all_gt = np.array(all_gt)

fpr, tpr, _ = roc_curve(all_gt, all_scores)
roc_auc = auc(fpr, tpr)
prec, rec, _ = precision_recall_curve(all_gt, all_scores)
pr_auc = auc(rec, prec)
print(f'AUC = {roc_auc:.4f}   AP = {pr_auc:.4f}')


# --------------------------------------------------------------------------- #
# FIGURE 1 — ROC curve
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color=BLUE, lw=3, label=f'VERA (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], '--', color=GREY, lw=2, label='Random (AUC = 0.500)')
ax.fill_between(fpr, tpr, alpha=0.12, color=BLUE)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — UCF-Crime Anomaly Detection')
ax.legend(loc='lower right', frameon=True)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
fig.savefig(f'{FIG_DIR}/1_roc_curve.png', bbox_inches='tight')
plt.close(fig)


# --------------------------------------------------------------------------- #
# FIGURE 2 — Precision-Recall curve
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(rec, prec, color=GREEN, lw=3, label=f'VERA (AP = {pr_auc:.3f})')
ax.fill_between(rec, prec, alpha=0.12, color=GREEN)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve — UCF-Crime')
ax.legend(loc='upper right', frameon=True)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
fig.savefig(f'{FIG_DIR}/2_precision_recall.png', bbox_inches='tight')
plt.close(fig)


# --------------------------------------------------------------------------- #
# FIGURE 3 — Comparison bar chart vs other methods (paper numbers)
# --------------------------------------------------------------------------- #
methods = ['SVM\nBaseline', 'Sultani\net al.', 'GCN\nAnomaly', 'LAVAD', 'VERA\n(ours)']
aucs    = [50.0, 75.4, 82.1, 80.3, roc_auc * 100]
colors  = [GREY, GREY, GREY, GREY, RED]

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar(methods, aucs, color=colors, edgecolor='black', linewidth=0.6)
for b, v in zip(bars, aucs):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f'{v:.1f}',
            ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('AUC (%)')
ax.set_title('AUC Comparison on UCF-Crime')
ax.set_ylim(0, 100)
ax.axhline(50, ls='--', color=GREY, alpha=0.5)
fig.savefig(f'{FIG_DIR}/3_method_comparison.png', bbox_inches='tight')
plt.close(fig)
# NOTE: the non-VERA numbers above are illustrative reference points.
# Replace them with the exact values from the paper's table for your slides.


# --------------------------------------------------------------------------- #
# FIGURE 4 — Score-vs-ground-truth timeline for a few example videos
# --------------------------------------------------------------------------- #
def plot_timeline(ax, key, fs, gt):
    t = np.arange(len(fs)) / FPS
    norm = (fs - fs.min()) / (fs.max() - fs.min() + 1e-8)
    ax.plot(t, norm, color=BLUE, lw=2, label='VERA anomaly score')
    ax.fill_between(t, 0, gt, color=RED, alpha=0.20, step='pre',
                    label='Ground-truth anomaly')
    ax.axhline(THRESHOLD, ls='--', color=GREY, lw=1.5, label=f'Threshold ({THRESHOLD})')
    ax.set_title(key)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Score')
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9, loc='upper right')

# pick 2 anomaly examples + 1 normal example if available
anomaly_examples = [(k, fs, gt) for (k, fs, gt) in per_video if gt.max() > 0][:2]
normal_examples  = [(k, fs, gt) for (k, fs, gt) in per_video if gt.max() == 0][:1]
examples = anomaly_examples + normal_examples

if examples:
    fig, axes = plt.subplots(len(examples), 1, figsize=(11, 3.2 * len(examples)))
    if len(examples) == 1:
        axes = [axes]
    for ax, (k, fs, gt) in zip(axes, examples):
        plot_timeline(ax, k, fs, gt)
    fig.suptitle('Frame-level Anomaly Detection — Score vs Ground Truth',
                 fontsize=16, fontweight='bold', y=1.005)
    fig.savefig(f'{FIG_DIR}/4_detection_timeline.png', bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# FIGURE 5 — Score distribution: normal vs anomalous frames
# --------------------------------------------------------------------------- #
anom_scores = all_scores[all_gt == 1]
norm_scores = all_scores[all_gt == 0]

fig, ax = plt.subplots(figsize=(8, 6))
bins = np.linspace(0, all_scores.max() + 1e-6, 40)
ax.hist(norm_scores, bins=bins, alpha=0.6, color=BLUE, label='Normal frames', density=True)
ax.hist(anom_scores, bins=bins, alpha=0.6, color=RED, label='Anomalous frames', density=True)
ax.set_xlabel('Anomaly score')
ax.set_ylabel('Density')
ax.set_title('Score Distribution: Normal vs Anomalous Frames')
ax.legend(frameon=True)
fig.savefig(f'{FIG_DIR}/5_score_distribution.png', bbox_inches='tight')
plt.close(fig)


# --------------------------------------------------------------------------- #
# FIGURE 6 — Summary metrics card
# --------------------------------------------------------------------------- #
n_videos = len(per_video)
n_anom = sum(1 for _, _, gt in per_video if gt.max() > 0)
n_norm = n_videos - n_anom

fig = plt.figure(figsize=(10, 4))
gs = gridspec.GridSpec(1, 4, wspace=0.3)
cards = [
    ('AUC', f'{roc_auc*100:.1f}%', BLUE),
    ('Avg Precision', f'{pr_auc*100:.1f}%', GREEN),
    ('Test Videos', f'{n_videos}', RED),
    ('Anomaly / Normal', f'{n_anom} / {n_norm}', GREY),
]
for i, (label, value, color) in enumerate(cards):
    ax = fig.add_subplot(gs[0, i])
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor=color, alpha=0.10, edgecolor=color, lw=2))
    ax.text(0.5, 0.62, value, ha='center', va='center', fontsize=26,
            fontweight='bold', color=color, transform=ax.transAxes)
    ax.text(0.5, 0.28, label, ha='center', va='center', fontsize=13,
            transform=ax.transAxes)
fig.suptitle('VERA — UCF-Crime Results Summary', fontsize=16, fontweight='bold')
fig.savefig(f'{FIG_DIR}/6_summary_card.png', bbox_inches='tight')
plt.close(fig)


print('\nSaved figures to ./%s/' % FIG_DIR)
for f in sorted(os.listdir(FIG_DIR)):
    print('  -', f)
