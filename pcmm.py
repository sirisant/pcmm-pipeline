import argparse
import os
from parcellation import get_subjects, run_parcellation
from bandpass import run_filtering
from hilbert import run_hilbert
from phase2complex import run_phase2complex

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PCMM pipeline')
    parser.add_argument('dataset_dir', type=str,   help='Path to dataset eg. C:\\data\\ds006623')
    parser.add_argument('--lowcut',    type=float, default=0.03, help='default: 0.03 Hz')
    parser.add_argument('--highcut',   type=float, default=0.07, help='default: 0.07 Hz')
    parser.add_argument('--order',     type=int,   default=5, help='default: 5')
    parser.add_argument('--steps', nargs='+', default=['all'],
                        help='Steps to run: parcellation filtering hilbert phase2complex (default: all)')
    args = parser.parse_args()

    # Paths in the project
    project_dir = os.path.dirname(os.path.abspath(__file__))
    atlas_file  = os.path.join(project_dir, 'atlas', 'Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S2_3T_MNI152NLin2009cAsym_2mm.nii')

    # Path to chosen dataset
    data_dir   = os.path.join(args.dataset_dir, 'derivatives', 'fmriprep_output')
    output_dir = os.path.join(args.dataset_dir, 'derivatives')

    # Ændre så man ikke behøver at lave analyse på alle subjekter på en gang?
    subjects = get_subjects(data_dir)
    print(f"Found: {len(subjects)} subject(s): {subjects}")

    # Run scripts on data
    if 'all' in args.steps or 'parcellation' in args.steps:
        run_parcellation(data_dir, os.path.join(output_dir, 'parcellation'), atlas_file, subjects)

    if 'all' in args.steps or 'filtering' in args.steps:
        run_filtering(os.path.join(output_dir, 'parcellation'), os.path.join(output_dir, 'bandpass'), subjects, fmriprep_dir=data_dir, lowcut=args.lowcut, highcut=args.highcut, order=args.order)

    if 'all' in args.steps or 'hilbert' in args.steps:
        run_hilbert(os.path.join(output_dir, 'bandpass'), os.path.join(output_dir, 'hilbert'), subjects)

    if 'all' in args.steps or 'phase2complex' in args.steps:
        run_phase2complex(os.path.join(output_dir, 'hilbert'), os.path.join(output_dir, 'phase2complex'), subjects)