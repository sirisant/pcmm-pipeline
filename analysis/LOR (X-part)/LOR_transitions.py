import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
K=5
# Extract X_part
X_part = np.loadtxt(r"C:\Users\Siri_\NRU\fMRI_internship\Analyse\diametrical_clustering_output2\X_part_K5.txt")

# Extract meta-data
df = pd.read_csv(r"C:\Users\Siri_\NRU\fMRI_internship\Analyse\phase2complex\Z_schaefer100tian32_metadata.csv")
df['subject']   = df['filename'].str.extract(r'(sub-\d+)')
df['scan_name'] = df['filename'].str.extract(r'sub-\d+_task-(.+?)_schaefer')
df = df.reset_index(drop=True)

# Sanity check
assert X_part.shape[0] == df['n_timepoints'].sum()

# Split into scans
split_points = np.cumsum(df['n_timepoints'].values)[:-1]
scans = np.split(X_part, split_points)

# LOR/ROR tider
lor_ror_data = {
    'subject': ['sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07',
                'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15', 'sub-16',
                'sub-17', 'sub-18', 'sub-19', 'sub-20', 'sub-21', 'sub-22',
                'sub-23', 'sub-24', 'sub-25', 'sub-26', 'sub-27', 'sub-28',
                'sub-29', 'sub-30'],
    'LOR_tr': [1160, 1385, 1573, 1010, 898, 1385, 1085, 1310, 1573, 898,
               485, 2248, 1010, 1573, 898, 1985, 1310, 1310, 1310, 635,
               1573, 1010, 485, 1385, 1835, 1010],
    'ROR_tr': [673, 410, 935, 673, 935, 1348, 673, 1535, 1460, None,
               2135, 1760, 1535, None, 1348, 2023, 1160, None, 2023, None,
               2023, 1348, 1760, None, None, 785]
}

df_lor = pd.DataFrame(lor_ror_data)
print(df_lor)

TR=0.8
window_tr = int(120 / TR)  # = 150 TRs

# For hvert subjekt udskær 4 vinduer:
# PreLOR:  imagery-run-2[LOR_tr - 150 : LOR_tr]
# LOR:     imagery-run-2[LOR_tr : LOR_tr + 150]
# PreROR:  imagery-run-3[ROR_tr - 150 : ROR_tr]
# ROR:     imagery-run-3[ROR_tr : ROR_tr + 150]

subjects_analysis = [f'sub-{str(i).zfill(2)}' for i in range(2, 31) if i not in [8, 9, 10, 29]]

rows = []
for sub_idx, sub in enumerate(subjects_analysis):
    lor_row = df_lor[df_lor['subject'] == sub].iloc[0]
    lor_tr  = int(lor_row['LOR_tr'])
    ror_tr  = lor_row['ROR_tr']

    # Find imagery-run-2 og imagery-run-3 skanninger
    img2_idx = sub_idx * 6 + 1  # imagery-run-2
    img3_idx = sub_idx * 6 + 2  # imagery-run-3
    img2 = scans[img2_idx]
    img3 = scans[img3_idx]

    # Beregn FO for hvert vindue
    windows = {
        'PreLOR': img2[lor_tr - window_tr : lor_tr],
        'LOR':    img2[lor_tr : lor_tr + window_tr],
    }

    # Tilføj PreROR og ROR hvis ROR ikke er N/A
    if pd.notna(ror_tr):
        ror_tr = int(ror_tr)
        windows['PreROR'] = img3[ror_tr - window_tr : ror_tr]
        windows['ROR']    = img3[ror_tr : ror_tr + window_tr]

    for window_name, window_data in windows.items():
        if len(window_data) == window_tr:  # tjek at vinduet er fuldt
            for k in range(K):
                rows.append({
                    'subject': sub,
                    'window':  window_name,
                    'state':   f'state_{k+1}',
                    'FO':      np.sum(window_data == k) / window_tr
                })
        else:
            print(f"Warning: {sub} {window_name} vindue er for kort: {len(window_data)} TRs")

df_windows = pd.DataFrame(rows)
print(df_windows.head(10))
print(df_windows.shape)

# Linear Mixed Effects Model

# Permutation test med max-T justering
n_permutations = 10000

def permutation_test_maxT(df_windows, window_ref, window_test, K, n_permutations=10000):
    
    # Find subjekter der har BEGGE vinduer
    subs_ref  = set(df_windows[df_windows['window'] == window_ref]['subject'].unique())
    subs_test = set(df_windows[df_windows['window'] == window_test]['subject'].unique())
    subs_both = subs_ref & subs_test  # intersection
    print(f"Subjekter med begge vinduer ({window_ref} & {window_test}): {len(subs_both)}")
    
    # Filtrer til kun disse subjekter
    df_filtered = df_windows[
        df_windows['window'].isin([window_ref, window_test]) & 
        df_windows['subject'].isin(subs_both)
    ]

    # Observerede t-statistikker
    t_observed = np.zeros(K)
    for k in range(K):
        df_state = df_filtered[df_filtered['state'] == f'state_{k+1}']
        pre  = df_state[df_state['window'] == window_ref]['FO'].values
        post = df_state[df_state['window'] == window_test]['FO'].values
        t_observed[k], _ = stats.ttest_rel(pre, post)

    # Permutationer
    t_max_dist = np.zeros(n_permutations)
    for perm in range(n_permutations):
        t_perm = np.zeros(K)
        for k in range(K):
            df_state = df_filtered[df_filtered['state'] == f'state_{k+1}']
            pre  = df_state[df_state['window'] == window_ref]['FO'].values
            post = df_state[df_state['window'] == window_test]['FO'].values
            
            signs = np.random.choice([-1, 1], size=len(pre))
            diff_perm = signs * (post - pre)
            t_perm[k], _ = stats.ttest_1samp(diff_perm, 0)
        
        t_max_dist[perm] = np.max(np.abs(t_perm))

    # Korrigerede p-værdier
    p_corrected = np.array([
        np.mean(t_max_dist >= np.abs(t_observed[k])) for k in range(K)
    ])

    return t_observed, p_corrected

# PreLOR vs LOR
print("PreLOR vs LOR:")
t_lor, p_lor = permutation_test_maxT(df_windows, 'PreLOR', 'LOR', K, n_permutations)
for k in range(K):
    sig = '*' if p_lor[k] < 0.05 else ''
    print(f"  State {k+1}: t={t_lor[k]:.3f}, p_corrected={p_lor[k]:.4f} {sig}")

# PreROR vs ROR (kun subjekter med ROR data)
df_ror = df_windows[df_windows['window'].isin(['PreROR', 'ROR'])]
print("\nPreROR vs ROR:")
t_ror, p_ror = permutation_test_maxT(df_ror, 'PreROR', 'ROR', K, n_permutations)
for k in range(K):
    sig = '*' if p_ror[k] < 0.05 else ''
    print(f"  State {k+1}: t={t_ror[k]:.3f}, p_corrected={p_ror[k]:.4f} {sig}")