import numpy as np
import pandas as pd

# def partition_X_part(data_dir, output_dir)

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

# Is it possible to save the partition? Save as individual and then load all again when needed in other scripts?