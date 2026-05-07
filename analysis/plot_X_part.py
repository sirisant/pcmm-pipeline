# Partition of X_part gets it own script and then I can just load it as data for the function
import numpy as np

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

### PLOT ###
import os
import matplotlib.pyplot as plt

output_dir = r'C:\Users\Siri_\NRU\fMRI_internship\projekt\analysis\plots_X_part'
os.makedirs(output_dir, exist_ok=True)

TR = 0.8 # I have it saved in a header
K=3 # args.K
subjects=[f'sub-{str(i).zfill(2)}' for i in range(2, 31) if i not in [8, 9, 10, 29]]

for sub_idx, sub in enumerate(subjects):
    
    sub_dir = os.path.join(output_dir, sub)
    os.makedirs(sub_dir, exist_ok=True)
    
    for run_idx, task_name in enumerate(scan_names):
        
        target_idx = sub_idx * 6 + run_idx
        data_to_plot = scans[target_idx]
        time_min = (np.arange(len(data_to_plot)) * TR) / 60

        fig, axes = plt.subplots(K, 1, figsize=(15, 2*K), sharex=True)

        for k in range(K):
            active = (data_to_plot == k).astype(float)
            axes[k].step(time_min, active, where='post', color=f'C{k}', linewidth=1.5)
            axes[k].set_ylabel(f'State {k+1}', fontsize=10)
            axes[k].set_yticks([0, 1])
            axes[k].set_yticklabels(['off', 'on'])
            axes[k].grid(axis='x', linestyle='--', alpha=0.4)

        axes[-1].set_xlabel('Time (minutes)', fontsize=12)
        fig.suptitle(f'{sub}: {task_name}', fontsize=14)
        plt.tight_layout()

        out_file = os.path.join(sub_dir, f'{task_name}.png')
        plt.savefig(out_file)
        plt.close()
        print(f"Saved: {out_file}")

# Create function for it all and and to pcmm mother script analysis part
# def plot_X_part_subs(data_dir, output_dir, subjects, K, TR, scan_names,):