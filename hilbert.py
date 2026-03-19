import numpy as np
from scipy.signal import hilbert
import os
import glob

# Complete Hilbert tranformation and phase exstraction
def hilbert_phase_extract(signal):
    return np.angle(hilbert(signal))

def run_hilbert(data_dir, output_dir, subjects):
    """Run hilbert phase on filtered timeseries"""
    os.makedirs(output_dir, exist_ok=False)
    
    for sub in subjects:
        func_dir = os.path.join(data_dir, sub, 'func')
        
        files = sorted(glob.glob(os.path.join(func_dir,f'{sub}_*_schaefer100tian32_filtered.txt')))

        out_dir = os.path.join(output_dir, sub, 'func')
        os.makedirs(out_dir, exist_ok=False)

        for txt_file in files:
            filtered_timeseries = np.loadtxt(txt_file)  # (timepoints, 132)

            # Beregn instantan fase for hver ROI
            phase_timeseries = np.zeros_like(filtered_timeseries)
            for r in range(filtered_timeseries.shape[1]):
                phase_timeseries[:, r] = hilbert_phase_extract(filtered_timeseries[:, r])

            task = [part for part in os.path.basename(txt_file).split('_') if 'task' in part][0]
            run  = [part for part in os.path.basename(txt_file).split('_') if 'run' in part][0]

            out_file = os.path.join(out_dir, f'{sub}_{task}_{run}_schaefer100tian32_hilbert.txt')
            np.savetxt(out_file, phase_timeseries)
            print(f"Saved: {out_file} {phase_timeseries.shape}")