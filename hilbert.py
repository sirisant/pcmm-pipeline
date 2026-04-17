import numpy as np
from scipy.signal import hilbert
import os
import glob

# Complete Hilbert tranformation and phase exstraction
def hilbert_phase_extract(signal, pad):
    """Extract instantaneous phase theta_j from signal using the Hilbert transform.
    Signal is demeaned before transformation."""
    signal_demeaned = signal - np.mean(signal)  # subtract mean
    analytic_signal = hilbert(signal_demeaned)  # A_t = s(t) + i*H(s(t))
    theta_j = np.angle(analytic_signal) # instantaneous phase in radians [-π, π])
    return theta_j[pad:-pad]

def run_hilbert(data_dir, output_dir, subjects, atlas_name, pad):
    """Run hilbert phase on filtered timeseries"""
    os.makedirs(output_dir, exist_ok=True)
    
    for sub in subjects:
        func_dir = os.path.join(data_dir, sub, 'func')
        
        files = sorted(glob.glob(os.path.join(func_dir,f'{sub}_*_{atlas_name}_filtered.txt')))

        out_dir = os.path.join(output_dir, sub, 'func')
        os.makedirs(out_dir, exist_ok=True)

        for txt_file in files:
            filtered_timeseries = np.loadtxt(txt_file)  # (timepoints, 132)

            nt_total, nROIs = filtered_timeseries.shape
            nt_trimmed = nt_total - 2*pad

            # Calculate the instantaneous phase for each ROI timeseries
            phase_timeseries = np.zeros((nt_trimmed, nROIs))
            for r in range(nROIs):
                signal = filtered_timeseries[:, r] # Timeseries for each ROI (1D)
                phase_timeseries[:, r] = hilbert_phase_extract(signal,pad)

            task = [part for part in os.path.basename(txt_file).split('_') if 'task' in part][0]
            run  = [part for part in os.path.basename(txt_file).split('_') if 'run' in part][0]

            out_file = os.path.join(out_dir, f'{sub}_{task}_{run}_{atlas_name}_hilbert.txt')
            np.savetxt(out_file, phase_timeseries)
            print(f"Saved: {out_file} {phase_timeseries.shape}")
