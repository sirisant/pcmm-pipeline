import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img
import glob
import os
import json

# Get all subjects
def get_subjects(data_dir):
    """Find all subject folders"""
    subjects = sorted([
        os.path.basename(s)
        for s in glob.glob(os.path.join(data_dir, 'sub-*'))
        if os.path.isdir(s) # Pick only directories
    ])
    return subjects

# Extract Repetition Time (TR) 
def get_TR(bold_file):
    """Extract TR from BOLD .json sidecar file"""
    json_file = bold_file.replace('.nii.gz', '.json')
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"No json sidecar found: {json_file}")
    with open(json_file, 'r') as f:
        TR = json.load(f)['RepetitionTime']
    print(f"Repetition Time: {TR}s")
    return TR

def run_parcellation(data_dir, output_dir, atlas_file, subjects, atlas_name):
    """Run atlas parcellation on bold files. Assumes bold files in MNI space."""
    os.makedirs(output_dir, exist_ok=True) # Create relevant output directory

    # Resample atlas only once using the first BOLD file from the first subject
    ref_file = sorted(glob.glob(os.path.join(data_dir, subjects[0], 'func', '*_space-MNI152NLin2009cAsym_*_desc-preproc_bold.nii.gz')))[0]
    ref_img = nib.load(ref_file)
    atlas_img = nib.load(atlas_file)
    atlas_img = resample_to_img(atlas_img, ref_img, interpolation='nearest') # Ensures the correct atlas dimensions

    atlas_data = atlas_img.get_fdata()
    nx, ny, nz = atlas_data.shape
    atlas_1d = atlas_data.reshape(nx*ny*nz) # Flatten voxels in atlas
    nROIs = int(atlas_1d.max()) # Number of ROIs given in the chosen atlas (default: 100 cortical and 32 subcortical)

    for sub in subjects:
        func_dir = os.path.join(data_dir, sub, 'func') # Builds path to get into all 'func' subject folders
        
        files = sorted(glob.glob(os.path.join(func_dir, f'{sub}_*_space-MNI152NLin2009cAsym_*_desc-preproc_bold.nii.gz'))) # Finds all desc-preproc bold .nii files in MNI space (sorted)
        out_dir = os.path.join(output_dir, sub, 'func')
        os.makedirs(out_dir, exist_ok=True) # Creates output folders for each subject

        for bold_file in files:
            bold_data = nib.load(bold_file).get_fdata()

            nx, ny, nz, nt = bold_data.shape
            bold_2d = bold_data.reshape(nx*ny*nz, nt) # Flatten voxels in bold file (voxels, timepoints)

            ROI_timeseries = np.zeros((nt, nROIs)) # Create zero-matrix
            for r in range(1, nROIs + 1):
                    ROI_timeseries[:, r-1] = bold_2d[atlas_1d == r, :].mean(axis=0) # Average BOLD-timeseries for ROI r
            
            # Extract types of 'task' and 'run' from each file
            task = [part for part in os.path.basename(bold_file).split('_') if 'task' in part][0]
            run  = [part for part in os.path.basename(bold_file).split('_') if 'run' in part][0]

            # Save .txt file for each ROI timeseries for each task type and run number
            TR = get_TR(bold_file) # Get TR from .json file
            out_file = os.path.join(out_dir, f'{sub}_{task}_{run}_{atlas_name}_timeseries.txt')
            np.savetxt(out_file, ROI_timeseries, header=f'TR={TR}') # Save TR as header
            print(f"Saved: {out_file} {ROI_timeseries.shape}")