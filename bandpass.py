from scipy.signal import butter, filtfilt
import os
import glob
import numpy as np

# Function for filtering a signal given low and high cut for bandpass filter with Frequencies
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="bandpass")
    return b, a

# Apply Filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order, pad):
    data_padded = np.concatenate([data[1:pad+1][::-1], data, data[-pad-1:-1][::-1]])
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = filtfilt(b, a, data_padded, padtype=None)
    return y

def run_filtering(data_dir, output_dir, subjects, atlas_name, lowcut, highcut, order, pad): # Defaults: lowcut (0.03 Hz), highcut (0.07 Hz), order (5)
    """Run butter bandpass filtering on parcellated timeseries"""
    os.makedirs(output_dir, exist_ok=True)

    for sub in subjects:
        func_dir = os.path.join(data_dir, sub, 'func')
        
        files = sorted(glob.glob(os.path.join(func_dir, f'{sub}_*_{atlas_name}_timeseries.txt')))

        out_dir = os.path.join(output_dir, sub, 'func')
        os.makedirs(out_dir, exist_ok=True)

        for txt_file in files:
            # Read TR from header
            with open(txt_file, 'r') as f:
                TR = float(f.readline().strip().replace('# TR=', ''))
            fs = 1 / TR

             # Load .txt fil
            ROI_timeseries = np.loadtxt(txt_file) 

            # Filter every timeseries
            nt, nROIs = ROI_timeseries.shape
            filtered_timeseries = np.zeros((nt+2*pad, nROIs))
            for r in range(nROIs):
                filtered_timeseries[:, r] = butter_bandpass_filter(ROI_timeseries[:, r], lowcut, highcut, fs, order, pad)

            task_name = [part for part in os.path.basename(txt_file).split('_') if 'task' in part][0] 
            run = [part for part in os.path.basename(txt_file).split('_') if 'run' in part][0]

            out_file = os.path.join(out_dir, f'{sub}_{task_name}_{run}_{atlas_name}_filtered.txt')
            np.savetxt(out_file, filtered_timeseries)
            print(f"Saved: {out_file} {filtered_timeseries.shape}")