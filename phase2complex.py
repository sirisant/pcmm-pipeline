import numpy as np
import os
import glob

def prepare_input(data_dir, output_dir, atlas_name):
    """Load hilbert phase timeseries and convert to complex eigenvectors.
    Returns concatenated complex matrix as input to the PCMM."""
    os.makedirs(output_dir, exist_ok=True)

    phase_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*', 'func', f'*{atlas_name}_hilbert.txt')))
    all_z = []
    for f in phase_files:
        phase = np.loadtxt(f)
        p = phase.shape[1]
        u_j = 1/np.sqrt(p) * np.exp(1j * phase) # e^(iθⱼ) on the unit circle
        all_z.append(u_j)

    Z = np.concatenate(all_z, axis=0)
    print(f"Input matrix shape: {Z.shape}")  # (n_total_timepoints, n_regions)

    out_file = os.path.join(output_dir, f'Z_{atlas_name}.npy')
    np.save(out_file, Z)
    print(f"Saved: {out_file}")
