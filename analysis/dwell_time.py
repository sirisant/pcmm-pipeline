import numpy as np

# Extract X_part
X_part = np.loadtxt(r"C:\Users\Siri_\NRU\fMRI_internship\Analyse\diametrical_clustering_output\X_part.txt")

# Number of subjects in the study
n_subjects = 25 # sub-29 removed

# Scans lengths for each of the 6 scans per subject
scan_lengths_per_subject = [1145, 2270, 2270, 1145, 770, 770]

# Scan names for the 6 scans
scan_names= ["imagery-run-1", "imagery-run-2", "imagery-run-3", "imagery-run-4", "rest-run-1", "rest-run-2"]

all_lengths = scan_lengths_per_subject * n_subjects

# Partition X_part along time axis into separate scans
split_points = np.cumsum(all_lengths)[:-1]
scans = np.split(X_part, split_points)  

# MEAN DWELL TIME 
# The time spent in a brain state before switching
# How much time is spent in a brain state at a time?
K = 3
TR = 0.8
def mean_dwell_time(scan, K, TR):
    """Calculate mean dwell time for each brain state in a scan.
    Excludes first and last episode of the entire scan (censored).
    Returns array of shape (K,) with mean dwell time in seconds."""
    
    episodes = [] 
    i = 0
    while i < len(scan):
        current_state = int(scan[i])
        j = i
        while j < len(scan) and scan[j] == current_state:
            j += 1
        episodes.append((current_state, j - i))
        i = j
    
    # Ekskluder første og sidste episode (censored)
    episodes = episodes[1:-1]
    
    # Beregn mean dwell time per state
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
print(dt_results)
print(dt_results.shape)  # (150, 3)

imagery1_dt = dt_results[0::6]   # alle imagery-run-1
imagery2_dt = dt_results[1::6]   # alle imagery-run-2
imagery3_dt = dt_results[2::6]   # alle imagery-run-3
imagery4_dt = dt_results[3::6]   # alle imagery-run-4
rest1_dt    = dt_results[4::6]   # alle rest-run-1
rest2_dt    = dt_results[5::6]   # alle rest-run-2

# STATISTICS

# Descriptive plot
import matplotlib.pyplot as plt

# Rest-1, Img-1, Img-2, Img-3, Img-4, Rest-2
new_order_indices = [4, 0, 1, 2, 3, 5]
new_labels = [scan_names[i] for i in new_order_indices]

fig, axes = plt.subplots(1, K, figsize=(15, 5))

for k in range(K):
    raw_data = [dt_results[i::6, k] for i in new_order_indices]
    
    axes[k].boxplot([d[~np.isnan(d)] for d in raw_data], labels=new_labels, showfliers=False)
    
    for i, scan_data in enumerate(raw_data):
        
        clean_data = scan_data[~np.isnan(scan_data)]
        x_jitter = np.random.normal(i + 1, 0.04, size=len(clean_data))
        axes[k].scatter(x_jitter, clean_data, alpha=0.5, s=15, color='darkred', zorder=3)

    axes[k].set_title(f'State {k+1}', fontweight='bold')
    axes[k].set_ylabel('Mean Dwell Time (s)')
    axes[k].tick_params(axis='x', rotation=45)
    axes[k].grid(axis='y', linestyle='--', alpha=0.3)

plt.suptitle('Mean Dwell Time per state', fontsize=14)
plt.tight_layout()
plt.show()