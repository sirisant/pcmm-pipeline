import numpy as np
import os
import glob

def run_phase2complex(data_dir, output_dir, subjects):
    os.makedirs(output_dir, exist_ok=False)

    for sub in subjects:
        func_dir = os.path.join(data_dir, sub, 'func')

        files = sorted(glob.glob(os.path.join(func_dir, f'{sub}_*_schaefer100tian32_hilbert.txt')))

        out_dir = os.path.join(output_dir, sub, 'func')
        os.makedirs(out_dir, exist_ok=False)

        for phase_file in files:
            phase = np.loadtxt(phase_file) # Shape: (timepoints, 132)

            p = phase.shape[1] # Number of regions = 132

            # Convert to complex eigenvector
            z = 1/np.sqrt(p) * np.exp(1j * phase)  # (timepoints, 132)

            task = [part for part in os.path.basename(phase_file).split('_') if 'task' in part][0]
            run  = [part for part in os.path.basename(phase_file).split('_') if 'run' in part][0]

            out_file = os.path.join(out_dir, f'{sub}_{task}_{run}_schaefer100tian32_phase2complex.npy')
            np.save(out_file, z)
            print(f"Saved: {out_file} {z.shape}")