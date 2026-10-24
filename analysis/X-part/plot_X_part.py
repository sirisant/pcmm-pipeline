# Partition of X_part gets it own script and then I can just load it as data for the function?
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Extract X_part
X_part = np.loadtxt(r"C:\Users\Siri_\NRU\fMRI_internship\Analyse\diametrical_clustering_output2\X_part_K5.txt")
print(X_part.shape)

# Extract meta-data
df = pd.read_csv(r"C:\Users\Siri_\NRU\fMRI_internship\Analyse\phase2complex\Z_schaefer100tian32_metadata.csv")

split_points = np.cumsum(df['n_timepoints'].values)[:-1]
scans = np.split(X_part, split_points)

print(f"Number of scans: {len(scans)}")       # 149
print(f"First scan: {scans[0].shape}")    # (1145,)

# PLOT 

# Parse subject and task-run label directly from filenames in the metadata
df['subject'] = df['filename'].str.extract(r'(sub-\d+)')
df['scan_name'] = df['filename'].str.extract(r'sub-\d+_task-(.+?)_schaefer')

# Build an index so each row knows which scan index it is globally
df = df.reset_index(drop=True)  # global scan index = df.index

TR = 0.8 # extract from header
K = 5 # use args.K
output_dir = r'C:\Users\Siri_\NRU\fMRI_internship\projekt\analysis\plots_X_part_K5'
os.makedirs(output_dir, exist_ok=True)

for sub, sub_df in df.groupby('subject', sort=False):

    sub_dir = os.path.join(output_dir, sub)
    os.makedirs(sub_dir, exist_ok=True)

    for _, row in sub_df.iterrows():

        scan_idx  = row.name          # global row index = position in `scans`
        task_name = row['scan_name']  # e.g. "imagery-run-2"
        data_to_plot = scans[scan_idx]
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
# def plot_X_part_subs(data_dir, output_dir, subjects, K, TR, scan_names,...):