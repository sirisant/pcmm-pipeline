import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# MEAN DWELL TIME
K  = 5 # use args.K
TR = 0.8 # extract from header

def mean_dwell_time(scan, K, TR):
    episodes = []
    i = 0
    while i < len(scan):
        current_state = int(scan[i])
        j = i
        while j < len(scan) and scan[j] == current_state:
            j += 1
        episodes.append((current_state, j - i))
        i = j

    episodes = episodes[1:-1]  # exclude first and last (censored)

    dwell_times = {k: [] for k in range(K)}
    for state, length in episodes:
        dwell_times[state].append(length)

    mean_dt = np.zeros(K)
    for k in range(K):
        if len(dwell_times[k]) > 0:
            mean_dt[k] = np.mean(dwell_times[k]) * TR
        else:
            mean_dt[k] = np.nan

    return mean_dt

dt_results = np.array([mean_dwell_time(scan.astype(int), K, TR) for scan in scans])
print(dt_results.shape)

# Ordered scan names: rest-run-1, imagery-run-1..4, rest-run-2
all_scan_names  = df['scan_name'].unique().tolist()
imagery_scans   = sorted([s for s in all_scan_names if 'imagery' in s])
rest_scans      = sorted([s for s in all_scan_names if 'rest' in s])
ordered_scan_names = [rest_scans[0]] + imagery_scans + [rest_scans[1]]

# PLOT
fig, axes = plt.subplots(1, K, figsize=(15, 5))

for k in range(K):
    data   = []
    labels = []
    for scan_name in ordered_scan_names:
        indices    = df.index[df['scan_name'] == scan_name].tolist()
        scan_data  = dt_results[indices, k]
        data.append(scan_data[~np.isnan(scan_data)])
        labels.append(scan_name)

    axes[k].boxplot(data, labels=labels, showfliers=False)

    for i, scan_data in enumerate(data):
        x_jitter = np.random.normal(i + 1, 0.04, size=len(scan_data))
        axes[k].scatter(x_jitter, scan_data, alpha=0.5, s=15, color='darkred', zorder=3)

    axes[k].set_title(f'State {k+1}', fontweight='bold')
    axes[k].set_ylabel('Mean Dwell Time (s)')
    axes[k].tick_params(axis='x', rotation=45)
    axes[k].grid(axis='y', linestyle='--', alpha=0.3)

plt.suptitle('Mean Dwell Time per state', fontsize=14)
plt.tight_layout()
plt.show()