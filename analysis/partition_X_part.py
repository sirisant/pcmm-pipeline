import numpy as np
import os

# def partition_X_part(data_dir, output_dir)

# Extract X_part
X_part = np.loadtxt(r"C:\Users\Siri_\NRU\fMRI_internship\projekt\diametrical_clustering_output\X_part.txt")
print(X_part.shape)

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

print(f"Number of scans: {len(scans)}")       # 150
print(f"First scan: {scans[0].shape}")    # (1145,)

# Is it possible to save the partition? Save as individual and then load all again when needed in other scripts?