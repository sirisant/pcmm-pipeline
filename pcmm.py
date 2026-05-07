import argparse
import os
from parcellation import get_subjects, run_parcellation
from bandpass import run_filtering
from hilbert import run_hilbert
from phase2complex import prepare_input
from diametrical_clustering import run_modeling

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PCMM pipeline')
    parser.add_argument('dataset_dir', type=str,   help='Path to dataset eg. C:\\data\\ds006623\\derivaties\\fmriprep_output')
    parser.add_argument('output_dir', type=str,   help='Path to dataset eg. C:\\data\\ds006623\\derivaties')
    parser.add_argument('atlas', type=str, help='Full path to atlas file eg. C:\\atlas\\Schaefer100.nii')
    parser.add_argument('--atlas_name', type=str, default='atlas', help='Short name for atlas for file names eg. schaefer100tian32')
    parser.add_argument('--lowcut',    type=float, default=0.03, help='default: 0.03 Hz')
    parser.add_argument('--highcut',   type=float, default=0.07, help='default: 0.07 Hz')
    parser.add_argument('--order',     type=int,   default=5, help='default: 5')
    parser.add_argument('--padding',     type=int,   default=100, help='Padding length (default: 100)')
    parser.add_argument('--K', type=int, default=3, help='Number of clusters to be inferred (default: 3)')
    parser.add_argument('--num_repl', type=int, default=1, help='Number of K-means replications to choose the best model from (default: 1)')
    parser.add_argument('--max_iter', type=int, default=10000, help='Max iterations (default: 10000)')
    parser.add_argument('--tol', type=float, default=1e-10, help='Stopping tolerance for changes in the objective function between consecutive iterations (default: 1e-10)')
    parser.add_argument('--init', type=str, default='++', choices=['++', 'uniform'], help='Initialization using the ++-algorithm or random values (default: ++)')
    parser.add_argument('--steps', nargs='+', default=['all'],
                        help='Steps to run: parcellation filtering hilbert phase2complex diametrical_clustering (default: all)')
    args = parser.parse_args()

    # Paths in the project
    project_dir = os.path.dirname(os.path.abspath(__file__))
    atlas_file  = args.atlas

    # Path to chosen dataset
    data_dir   = args.dataset_dir
    # Path to chosen output folder
    output_dir = args.output_dir

    # Ændre så man ikke behøver at lave analyse på alle subjekter på en gang?
    subjects = get_subjects(data_dir)
    print(f"Found: {len(subjects)} subject(s): {subjects}")

    # Run scripts on data
    if 'all' in args.steps or 'parcellation' in args.steps:
        run_parcellation(data_dir, os.path.join(output_dir, 'parcellation'), atlas_file, subjects, args.atlas_name)

    if 'all' in args.steps or 'filtering' in args.steps:
        run_filtering(os.path.join(output_dir, 'parcellation'), os.path.join(output_dir, 'bandpass'), subjects, args.atlas_name, args.lowcut, args.highcut, args.order, args.padding)

    if 'all' in args.steps or 'hilbert' in args.steps:
        run_hilbert(os.path.join(output_dir, 'bandpass'), os.path.join(output_dir, 'hilbert'), subjects, args.atlas_name, args.padding)

    if 'all' in args.steps or 'phase2complex' in args.steps:
        prepare_input(os.path.join(output_dir, 'hilbert'), os.path.join(output_dir, 'phase2complex'), args.atlas_name)

    if 'all' in args.steps or 'diametrical_clustering' in args.steps:
        run_modeling(os.path.join(output_dir, 'phase2complex'), os.path.join(output_dir, 'diametrical_clustering'), args.K, args.num_repl, args.max_iter, args.tol, args.init, args.atlas_name)

## Visualization and analysis

    #if 'partition_X_part' in args.steps:
            #partition(os.path.join(output_dir, 'diametrical_clustering'), os.path.join(output_dir, 'partition_X_part'))

    #if 'plot_X_part' in args.steps:
        #plot_X_part_subs()