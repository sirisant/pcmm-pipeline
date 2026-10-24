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

# Sanity check
assert X_part.shape[0] == df['n_timepoints'].sum()

# Split into scans
split_points = np.cumsum(df['n_timepoints'].values)[:-1]
scans = np.split(X_part, split_points)

# FRACTIONAL OCCURRENCE
K = 5 # use args.K
fo_results = []
for scan in scans:
    n_tp = len(scan)
    fo = np.array([np.sum(scan == k) / n_tp for k in range(K)])
    fo_results.append(fo)

fo_results = np.array(fo_results)  # (n_scans, K)
assert np.allclose(fo_results.sum(axis=1), 1.0)

# Unique scan names in a desired order - should be more dynamic?
all_scan_names = df['scan_name'].unique().tolist()
imagery_scans  = sorted([s for s in all_scan_names if 'imagery' in s])
rest_scans     = sorted([s for s in all_scan_names if 'rest' in s])
ordered_scan_names = [rest_scans[0]] + imagery_scans + [rest_scans[1]]

# PLOT
fig, axes = plt.subplots(1, K, figsize=(15, 5), sharey=True)

for k in range(K):
    data   = []
    labels = []
    for scan_name in ordered_scan_names:
        indices = df.index[df['scan_name'] == scan_name].tolist()
        data.append(fo_results[indices, k])
        labels.append(scan_name)

    axes[k].boxplot(data, labels=labels, showfliers=False)

    for i, scan_data in enumerate(data):
        x_jitter = np.random.normal(i + 1, 0.04, size=len(scan_data))
        axes[k].scatter(x_jitter, scan_data, alpha=0.4, s=10, color='darkblue', zorder=3)

    axes[k].set_title(f'State {k+1}', fontweight='bold')
    axes[k].tick_params(axis='x', rotation=45)
    axes[k].set_ylim(-0.05, 1.05)
    axes[k].grid(axis='y', linestyle='--', alpha=0.3)

plt.suptitle('Fractional Occurrence', fontsize=14)
plt.tight_layout()
plt.show()