import json
import pathlib
from sklearn.metrics import auc, roc_curve
import numpy as np
import glob
import os

# ---------------------------------------------------------------------------
# Paths — edit these if your data lives elsewhere.
# On Kaggle you may want absolute paths like:
#   ann_root      = '/kaggle/working/VERA/Data/UCF_Eval.json'
#   vision_folder = '/kaggle/working/Data/vision_features/'
#   score_folder  = '/kaggle/working/Data/segment_level_score/'
# ---------------------------------------------------------------------------
ann_root      = 'Data/UCF_Eval.json'
vision_folder = 'Data/vision_features/'
score_folder  = 'Data/segment_level_score/'

# Video frame rate (UCF-Crime frames were extracted at 30 fps).
FPS = 30
# A frame is called "suspicious" when its smoothed score is above this.
THRESHOLD = 0.5
# Where the per-video detection report is written.
REPORT_PATH = 'anomaly_report.json'


def gaussian_kernel_original(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaussian_smoothing(data, sigma):
    n = len(data)
    x = np.arange(n)
    centroid_index = int(n / 2)
    kernel_values = gaussian_kernel_original(x, centroid_index, sigma)
    return kernel_values * data


def gaussian_kernel(size, sigma):
    kernel = np.exp(-np.linspace(-size // 2, size // 2, size) ** 2 / (2 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_smooth_1d(data, size=5, sigma=1.0):
    kernel = gaussian_kernel(size, sigma)
    return np.convolve(data, kernel, mode='same')


def softmax(row_vector):
    exp_row = np.exp(row_vector - np.max(row_vector))
    return exp_row / np.sum(exp_row)


def frames_to_ranges(mask):
    """Turn a boolean per-frame mask into a list of (start_frame, end_frame) ranges."""
    ranges = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            ranges.append((start, i - 1))
            start = None
    if start is not None:
        ranges.append((start, len(mask) - 1))
    return ranges


# ---------------------------------------------------------------------------
# Load any extra scores that live under ./scores_77/tests/ (optional).
# ---------------------------------------------------------------------------
result = {}
folder_path = './scores_77/tests/'
for file in glob.glob(os.path.join(folder_path, '**', '*.json'), recursive=True):
    file_name = os.path.basename(file)
    with open(file, 'r') as f:
        result[file_name[:-5]] = json.load(f)

all_predict_score = []
all_gt = []
report = {}          # per-video suspicious ranges go here

data_path = pathlib.Path(ann_root)
with data_path.open(encoding='utf-8') as f:
    annotation = json.load(f)

for v_i in range(len(annotation)):
    sample = annotation[v_i]
    key = sample['video'].split('/')[-1].split('.')[0]

    # segment-level 0/1 scores produced by the VLM
    score_path = score_folder + key + '.json'
    with open(score_path, 'r') as f:
        score = json.load(f)
    result[key] = score

    # vision similarity features (ImageBind)
    v_array = np.load(vision_folder + key + '.npy')
    v_norms = np.linalg.norm(v_array, axis=1, keepdims=True)
    normalized_v = v_array / v_norms
    v_array = np.dot(normalized_v, normalized_v.T)

    num_segment = v_array.shape[0]
    top_n = int(0.15 * num_segment)
    v_indices = np.argsort(v_array, axis=1)[:, -top_n:]
    v_top_values = np.take_along_axis(v_array, v_indices, axis=1)

    pred_score = []
    sampling_rate = 16
    start_index = [k for k in result[key].keys()]

    all_score = [result[key][start]['score'] for start in start_index]

    # Step 2: refine each segment using its visually-similar neighbours
    score_all = []
    for i, start in enumerate(start_index):
        neighbor_score = np.array([all_score[n] for n in v_indices[i].tolist()])
        v_softmax_values = softmax(v_top_values[i] * 10)
        v_segment_score = np.dot(v_softmax_values, neighbor_score)
        score_all.append(v_segment_score)

    smoothed_data = gaussian_smooth_1d(np.array(score_all), 15, 10)

    # Step 3: expand segment scores back to per-frame scores
    for i, start in enumerate(start_index):
        segment_score = round(smoothed_data[i], 1)
        if i != len(start_index) - 1:
            pred_score.extend([segment_score] * sampling_rate)
        else:
            num_ele = int(result[key][start]['end'] - int(start))
            pred_score.extend([segment_score] * num_ele)

    sigma = int(len(pred_score) / 2)
    pred_score = gaussian_smoothing(np.array(pred_score), sigma)

    # ------------------------------------------------------------------
    # NEW: keep the frame-level info instead of throwing it away.
    # ------------------------------------------------------------------
    mask = pred_score > THRESHOLD
    ranges = frames_to_ranges(mask)
    suspicious = [
        {
            'start_frame': int(s),
            'end_frame': int(e),
            'start_sec': round(s / FPS, 1),
            'end_sec': round(e / FPS, 1),
            'peak_score': round(float(pred_score[s:e + 1].max()), 3),
        }
        for (s, e) in ranges
    ]
    report[key] = {
        'num_frames': int(len(pred_score)),
        'max_score': round(float(pred_score.max()), 3),
        'suspicious_ranges': suspicious,
    }

    if suspicious:
        spans = ', '.join(f"{r['start_sec']}s-{r['end_sec']}s" for r in suspicious)
        print(f"[ANOMALY] {key}: {spans}")
    else:
        print(f"[  ok   ] {key}: no suspicious frames")

    # build ground truth for AUC
    gt = [0.0] * annotation[v_i]['length']
    for anno_i in range(0, len(annotation[v_i]['temporal_label']), 2):
        if annotation[v_i]['temporal_label'][anno_i] != -1:
            anno_s = annotation[v_i]['temporal_label'][anno_i]
            anno_e = min(annotation[v_i]['temporal_label'][anno_i + 1], annotation[v_i]['length'])
            gt[anno_s:anno_e] = [1.0] * (anno_e - anno_s)

    all_predict_score.extend(pred_score)
    all_gt.extend(gt)

# ---------------------------------------------------------------------------
# Overall AUC (same number as the original script) + save the report.
# ---------------------------------------------------------------------------
fpr, tpr, _ = roc_curve(all_gt, all_predict_score)
roc_auc = auc(fpr, tpr)

with open(REPORT_PATH, 'w') as f:
    json.dump(report, f, indent=2)

print('\n' + '=' * 60)
print(f'Overall frame-level ROC-AUC : {roc_auc:.4f}')
print(f'Per-video detection report  : {REPORT_PATH}')
print('=' * 60)
