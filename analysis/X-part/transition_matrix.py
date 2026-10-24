import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Extract X_part
X_part = np.loadtxt(r"C:\Users\Siri_\NRU\fMRI_internship\Analyse\diametrical_clustering_output2\X_part_K5.txt")

# Extract meta-data
df = pd.read_csv(r"C:\Users\Siri_\NRU\fMRI_internship\Analyse\phase2complex\Z_schaefer100tian32_metadata.csv")
df['subject']   = df['filename'].str.extract(r'(sub-\d+)')
df['scan_name'] = df['filename'].str.extract(r'sub-\d+_task-(.+?)_schaefer')
df = df.reset_index(drop=True)

assert X_part.shape[0] == df['n_timepoints'].sum()

split_points = np.cumsum(df['n_timepoints'].values)[:-1]
scans = np.split(X_part, split_points)

# TRANSITION MATRIX
def _transition_matrix(labels, K):
    """
    Compute K×K Markov transition probability matrix from a state sequence.

    Parameters
    ----------
    labels : array-like, shape (T,)
        Integer state labels 0..K-1.
    K : int

    Returns
    -------
    P : np.ndarray, shape (K, K)
    """
       
    P = np.zeros((K, K))
    for t in range(len(labels) - 1):
        P[labels[t], labels[t + 1]] += 1
    row_sums = P.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        P = np.where(row_sums > 0, P / row_sums, 0.0)
    return P

K = 5 # use args.K
subjects = df['subject'].unique()
n_subjects = len(subjects)

# Ordered scan names: rest-run-1, imagery-run-1..4, rest-run-2
all_scan_names = df['scan_name'].unique().tolist()
imagery_scans  = sorted([s for s in all_scan_names if 'imagery' in s])
rest_scans     = sorted([s for s in all_scan_names if 'rest' in s])
ordered_scan_names = [rest_scans[0]] + imagery_scans + [rest_scans[1]]
n_scans = len(ordered_scan_names)

# Compute per-subject per-scan transition matrices
# Shape: (n_subjects, n_scans, K, K) - NaN matrix where scan is missing
matrices = np.full((n_subjects, n_scans, K, K), np.nan)

for i, sub in enumerate(subjects):
    sub_df = df[df['subject'] == sub]
    for t, scan_name in enumerate(ordered_scan_names):
        row = sub_df[sub_df['scan_name'] == scan_name]
        if row.empty:
            continue  # missing scan (e.g. sub-22 rest-run-1) stays NaN
        scan_idx = row.index[0]
        labels = scans[scan_idx].astype(int)
        matrices[i, t] = _transition_matrix(labels, K)

# Group-level stats (nanmean/nanstd handles missing scans)
mean_matrices = np.nanmean(matrices, axis=0)  # (n_scans, K, K)
std_matrices  = np.nanstd(matrices, axis=0)

# PLOT
n_cols = 3
n_rows = int(np.ceil(n_scans / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes = axes.flatten()

for i, scan_name in enumerate(ordered_scan_names):
    annot_labels = [
        [f"{m:.2f}\n±{s:.2f}" for m, s in zip(row_m, row_s)]
        for row_m, row_s in zip(mean_matrices[i], std_matrices[i])
    ]
    sns.heatmap(mean_matrices[i],
                annot=annot_labels,
                fmt="",
                cmap="YlGnBu",
                cbar=True,
                square=True,
                xticklabels=[f'T S{k+1}' for k in range(K)],
                yticklabels=[f'F S{k+1}' for k in range(K)],
                annot_kws={"size": 8},
                ax=axes[i])
    axes[i].set_title(scan_name, fontweight='bold', size=12, pad=10)

# Hide any unused subplots
for j in range(n_scans, len(axes)):
    axes[j].set_visible(False)

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.88, wspace=0.4, hspace=0.4)
plt.suptitle(f'Mean Transition Probabilities ± SD (n={n_subjects})', fontsize=16, y=0.96)
plt.show()