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

# Ordered scan names: rest-run-1, imagery-run-1..4, rest-run-2
all_scan_names = df['scan_name'].unique().tolist()
imagery_scans  = sorted([s for s in all_scan_names if 'imagery' in s])
rest_scans     = sorted([s for s in all_scan_names if 'rest' in s])
ordered_scan_names = [rest_scans[0]] + imagery_scans + [rest_scans[1]]

# SWITCHING FREQUENCY
TR = 0.8 # extract from header

def get_switching_freq_hz(scan, tr):
    n_switches = np.sum(scan[:-1] != scan[1:])
    total_time_seconds = len(scan) * tr
    return n_switches / total_time_seconds

sf_hz_results = np.array([get_switching_freq_hz(scan, TR) for scan in scans])

# PLOT
plot_data = []
for scan_name in ordered_scan_names:
    indices = df.index[df['scan_name'] == scan_name].tolist()
    plot_data.append(sf_hz_results[indices])

fig, ax = plt.subplots(figsize=(10, 6))

bp = ax.boxplot(plot_data, labels=ordered_scan_names, showfliers=False, patch_artist=True)
for box in bp['boxes']:
    box.set(facecolor='plum', alpha=0.4)

for i, run_data in enumerate(plot_data):
    x_jitter = np.random.normal(i + 1, 0.04, size=len(run_data))
    ax.scatter(x_jitter, run_data, alpha=0.7, s=25, color='purple', edgecolor='white', zorder=3)

ax.set_title('Global Switching Frequency (Hz)', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency (switches / second)')
ax.set_xlabel('Scan Run')
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()