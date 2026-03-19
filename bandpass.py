from scipy.signal import butter, filtfilt
import os
import glob
import numpy as np
import json

# Extract Repetition Time (TR) from first file (assume it's the same for all tasks and runs)
def get_TR(data_dir, subjects):
    """Extract TR from first BOLD .json file from the first subject"""
    json_file = sorted(glob.glob(os.path.join(data_dir, subjects[0], 'func', '*_desc-preproc_bold.json')))[0]
    with open(json_file, 'r') as f:
        TR = json.load(f)['RepetitionTime']
    print(f"Repetition Time: {TR}s")
    return TR

# Function for filtering a signal given low and high cut for bandpass filter with Frequencies
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="bandpass")
    return b, a

# Apply Filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def run_filtering(data_dir, output_dir, subjects, fmriprep_dir, lowcut, highcut, order): # Defaults: lowcut (0.03 Hz), highcut (0.07 Hz), order (5)
    """Run butter bandpass filtering on parcellated timeseries"""
    os.makedirs(output_dir, exist_ok=False)

    TR = get_TR(fmriprep_dir, subjects) # Extract Repetition Time from .json file
    fs = 1 / TR

    for sub in subjects:
        func_dir = os.path.join(data_dir, sub, 'func')
        
        files = sorted(glob.glob(os.path.join(func_dir, f'{sub}_*_schaefer100tian32_timeseries.txt')))

        out_dir = os.path.join(output_dir, sub, 'func')
        os.makedirs(out_dir, exist_ok=False)

        for txt_file in files:
            ROI_timeseries = np.loadtxt(txt_file)  # Load .txt fil

            # Filter every timeseries
            filtered_timeseries = np.zeros_like(ROI_timeseries)
            for r in range(ROI_timeseries.shape[1]):
                filtered_timeseries[:, r] = butter_bandpass_filter(ROI_timeseries[:, r], lowcut=lowcut, highcut=highcut, fs=fs, order=order)

            task_name = [part for part in os.path.basename(txt_file).split('_') if 'task' in part][0] 
            run = [part for part in os.path.basename(txt_file).split('_') if 'run' in part][0]

            out_file = os.path.join(out_dir, f'{sub}_{task_name}_{run}_schaefer100tian32_filtered.txt')
            np.savetxt(out_file, filtered_timeseries)
            print(f"Saved: {out_file} {filtered_timeseries.shape}")