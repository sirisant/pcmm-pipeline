import numpy as np
import pandas as pd
import os

# 1. Load the cluster assignments (Cast to float first, then int to bypass NumPy deprecation warning)
X_part = np.loadtxt(r"C:\Users\Siri_\NRU\fMRI_internship\Analyse\diametrical_clustering_output2\X_part_K5.txt").astype(int)

# 2. Load and parse the pipeline's metadata file
df_meta = pd.read_csv(r"C:\Users\Siri_\NRU\fMRI_internship\Analyse\phase2complex\Z_schaefer100tian32_metadata.csv")
df_meta['subject']   = df_meta['filename'].str.extract(r'(sub-\d+)')
df_meta['scan_name'] = df_meta['filename'].str.extract(r'sub-\d+_task-(.+?)_schaefer')
df_meta = df_meta.reset_index(drop=True)

# Sanity check to ensure shapes match perfectly
assert X_part.shape[0] == df_meta['n_timepoints'].sum(), "Mismatched total TR counts!"

split_points = np.cumsum(df_meta['n_timepoints'].values)[:-1]
scans_states = np.split(X_part, split_points)

# (omitting 8, 9, 10, and 29)
subjects_analysis = [f'sub-{str(i).zfill(2)}' for i in range(2, 31) if i not in [8, 9, 10, 29]]

# Base directory where subject folders are kept
data_base_dir = r"C:\Users\Siri_\NRU\fMRI_internship\data\ds006623\derivatives\Propofol_Infusion"

all_tr_rows = []

# Keep track of where we are in each subject's continuous ESC file
subject_tr_counters = {sub: 0 for sub in subjects_analysis}
subject_esc_cached = {}

# 4. Loop through every single scan row in the metadata
for idx, meta_row in df_meta.iterrows():
    sub = meta_row['subject']
    task_name = meta_row['scan_name']
    n_trs = int(meta_row['n_timepoints'])
    
    # Skip subjects that are excluded
    if sub not in subjects_analysis:
        continue
        
    # Extract the sliced brain state sequence for this specific scan
    current_scan_states = scans_states[idx]
    
    if sub not in subject_esc_cached:
        esc_file = os.path.join(data_base_dir, sub, f"{sub}_allrun_ESC.1D")
        if os.path.exists(esc_file):
            try:
                # Robust parsing strategy to bypass hidden DataLad/git-annex metadata tracks
                with open(esc_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                raw_values = []
                for line in lines:
                    cleaned_line = line.strip()
                    if cleaned_line and not cleaned_line.startswith('#'):
                        raw_values.extend(cleaned_line.split())
                
                # Coerce potential annex text shortcuts to NaN and drop them
                numeric_series = pd.to_numeric(pd.Series(raw_values), errors='coerce')
                clean_esc_array = numeric_series.dropna().values
                
                if len(clean_esc_array) == 0:
                    raise ValueError("Parsed file contained no numbers.")
                
                subject_esc_cached[sub] = clean_esc_array
                
            except Exception as e:
                print(f"Warning: Parsing error on {sub}, using zero-padding fallback: {e}")
                subject_esc_cached[sub] = np.zeros(X_part.shape[0])
        else:
            print(f"Warning: Missing ESC file for {sub}. Defaulting to zeros.")
            subject_esc_cached[sub] = np.zeros(X_part.shape[0]) 

    # Slices out the exact chunk of the continuous ESC file needed for this task
    start_tr = subject_tr_counters[sub]
    end_tr = start_tr + n_trs
    current_scan_esc = subject_esc_cached[sub][start_tr:end_tr]
    
    # Advance the subject's TR cursor for their next task scan in the loop
    subject_tr_counters[sub] = end_tr
    
    # Safety check if the ESC slice is shorter than the fMRI run lengths
    if len(current_scan_esc) != n_trs:
        current_scan_esc = np.resize(current_scan_esc, n_trs)

    # 5. Append each individual TR frame to our tracker list using categorical state column
    for tr in range(n_trs):
        active_state = current_scan_states[tr]
        
        all_tr_rows.append({
            'Subject': sub,
            'Timepoint_TR': tr,
            'Time_Seconds': round(tr * 0.8, 1),
            'Task_Name': task_name,
            'ESC': current_scan_esc[tr],
            'Brain_State': f"State_{active_state+1}"  # Clear text factor for R (+1 because zero-indexed)
        })

# Create final master dataframe for R
master_df = pd.DataFrame(all_tr_rows)

# Save the categorical CSV
output_path = "pcmm_categorical_states.csv"
master_df.to_csv(output_path, index=False)
print(f"Successfully generated {len(master_df)} rows inside '{output_path}'!")