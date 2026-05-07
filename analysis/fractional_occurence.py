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

# FRACTIONAL OCCURENCE (FO)
# The fraction of time points in a scan session assigned to that brain state
# How much time is spent in each brain state?

K=3 # args.K
fo_results = []

for scan in scans:
    n_timepoints = len(scan)

    # Counts of each active state (0, 1, 2 for K=3) for all timepoints in a scan
    # Normalized by total number of timepoints in the scan
    fo = np.array([np.sum(scan == k) / n_timepoints for k in range(K)])
    fo_results.append(fo)

fo_results = np.array(fo_results)  # (n_scans, K)
print(fo_results.shape)  # (150, 3)

check_sums = np.sum(fo_results, axis=1)
print(check_sums) # Should be all 1's

# STATISTICS
# Start with tasks (imagery-runs) - are there differences in FO during tasks under different levels of anesthesia?

# Descriptive plot
import matplotlib.pyplot as plt

new_order = [4, 0, 1, 2, 3, 5]
new_scan_names = [scan_names[i] for i in new_order]

fig, axes = plt.subplots(1, K, figsize=(15, 5), sharey=True)

for k in range(K):
    data = [fo_results[i::6, k] for i in new_order]
    
    # Boxplot
    axes[k].boxplot(data, labels=new_scan_names, showfliers=False)
    
    # add datapoints with jitter
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