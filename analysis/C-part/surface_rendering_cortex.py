import numpy as np
import nibabel as nib
nib.imageglobals.logger.setLevel(40)
import matplotlib.pyplot as plt
import pandas as pd
import nilearn.plotting as plotting
import hcp_utils as hcp
from matplotlib.colors import LinearSegmentedColormap

# 1. Setup cyclic colormap for phase angles (-pi to pi wrapping)
angles = np.array([0, 0.25, 0.5, 0.75, 1.0])  # Normalized positions of -pi to pi
colors = ['blue', 'green', 'red', 'yellow', 'blue']  # Color for each angle
cyclic_cmap = LinearSegmentedColormap.from_list('cyclic_map', list(zip(angles, colors))) 

plt.figure(figsize=(6, 1))
plt.imshow([np.linspace(-1, 1, 256)], aspect='auto', cmap=cyclic_cmap)
plt.axis('off')
plt.title('Cyclic Colormap')
plt.show()

def load_mask_and_schaefer_parcellation(parcellation_file, label_file, mask_lh_file, mask_rh_file):
    """
    Import mask and parcellation data. (Exactly your supervisor's code)
    """
    parcel_labels = np.loadtxt(label_file, dtype=str, usecols=0)
    parcel_labels = parcel_labels[0::2] 

    # Parcellation-wise power
    mask_lh = np.loadtxt(mask_lh_file, dtype=bool)
    mask_rh = np.loadtxt(mask_rh_file, dtype=bool)
    parcellation = nib.load(parcellation_file).get_fdata()[0]
    parcellation_lh = parcellation[:mask_lh.sum()]
    parcellation_rh = parcellation[mask_lh.sum():mask_lh.sum()+mask_rh.sum()]
    parcellation_lh_expanded = np.zeros(mask_lh.shape)
    parcellation_lh_expanded[mask_lh] = parcellation_lh
    parcellation_rh_expanded = np.zeros(mask_rh.shape)
    parcellation_rh_expanded[mask_rh] = parcellation_rh

    return parcel_labels, parcellation, {'mask_lh':mask_lh, 'mask_rh':mask_rh}, {'parcellation_lh_expanded':parcellation_lh_expanded, 'parcellation_rh_expanded':parcellation_rh_expanded}

def plot_surf(spatial_map, output_file, vmin, vmax, parcellation_file, label_file, mask_lh_file, mask_rh_file, cmap='bwr'):
    parcel_labels, parcellation, masks, parcellation_extended = load_mask_and_schaefer_parcellation(parcellation_file, label_file, mask_lh_file, mask_rh_file)
    unique_parcels = np.unique(parcellation)
    unique_parcels = unique_parcels[unique_parcels != 0]  
    h2 = {'lh': 'left', 'rh': 'right'}
    h3 = {'lh': 'left', 'rh': 'right'}

    if not isinstance(spatial_map, pd.Series):
        spatial_map = pd.Series(spatial_map, index=parcel_labels)

    for hemi in ['lh', 'rh']:
        cortex_map = np.empty(masks['mask_'+hemi].shape)
        cortex_map.fill(np.nan)
        for roi_label, roi_number in zip(parcel_labels, unique_parcels):
            if hemi.upper() in roi_label:
                cortex_map[parcellation_extended['parcellation_'+hemi+'_expanded'] == roi_number] = spatial_map[roi_label]
        
        for view in ['lateral', 'medial']:
            plotting.plot_surf(hcp.mesh['midthickness_'+h3[hemi]], cortex_map, bg_map=hcp.mesh['sulc_'+h3[hemi]],
                               symmetric_cmap=None, bg_on_data=True, colorbar=True, 
                               cmap=cmap, vmin=vmin, vmax=vmax, alpha=1, hemi=h2[hemi], view=view,  # MINIMAL MODIFICATION: changed cmap='bwr' to cmap=cmap
                               output_file=output_file+'_'+hemi+'_'+view+'.png')

if __name__ == "__main__":
    C = np.loadtxt(r"C:\Users\Siri_\NRU\fMRI_internship\projekt\diametrical_clustering_output\K5\C_K5.txt", dtype=complex)
    vmin_real = -0.1
    vmax_real = 0.1
    
    parcellation_file = r"C:\Users\Siri_\NRU\fMRI_internship\atlas\Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S2.dlabel.nii"
    label_file = r"C:\Users\Siri_\NRU\fMRI_internship\atlas\Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S2_label.txt"
    
    mask_lh_file = r"C:\Users\Siri_\NRU\fMRI_internship\atlas\fsLR_32k_cortex-lh_mask.txt"
    mask_rh_file = r"C:\Users\Siri_\NRU\fMRI_internship\atlas\fsLR_32k_cortex-rh_mask.txt"

    for k in range(C.shape[0]):
        spatial_map = C[k]
        output_file = f'spectral_analysis/plots/anesthesia_component_{k+1}'
        
        # 1. Initialize full 132-element arrays containing NaN values
        full_real_map = np.full(132, np.nan)
        full_angle_map = np.full(132, np.nan)
        
        # 2. Assign only the cortical structures (indices 32 through 132)
        full_real_map[32:132] = np.real(spatial_map)[32:132]
        full_angle_map[32:132] = np.angle(spatial_map)[32:132]
            
        # A: Plot Real Part (Defaulting to 'bwr')
        plot_surf(full_real_map, output_file + '_real', vmin_real, vmax_real, 
                  parcellation_file, label_file, mask_lh_file, mask_rh_file, cmap='bwr')
            
        # B: Plot Phase Angle (Passing custom cyclic map)
        plot_surf(full_angle_map, output_file + '_angle', -np.pi, np.pi, 
                  parcellation_file, label_file, mask_lh_file, mask_rh_file, cmap=cyclic_cmap)
        

#x,y,z,u = load_mask_and_schaefer_parcellation(atlas, labels_img, mask_lh, mask_rh)
#print(x)
#print(y)
#print(z)
#print(u)

