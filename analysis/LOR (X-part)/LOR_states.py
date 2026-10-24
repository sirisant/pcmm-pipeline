from scipy import stats
import numpy as np
import pandas as pd

K = 5
TR = 0.8

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
subjects_analysis = [f'sub-{str(i).zfill(2)}' for i in range(2, 31) if i not in [8, 9, 10, 29]]

# Beregn FO for hver periode dynamisk
rows_periods = []
period_order = ['rest-1', 'pre-LOR', 'LOR', 'post-LOR', 'rest-2']

for sub in subjects_analysis:
    sub_df  = df[df['subject'] == sub]
    lor_row = df_lor[df_lor['subject'] == sub].iloc[0]
    lor_tr  = int(lor_row['LOR_tr'])
    ror_tr  = lor_row['ROR_tr']

    def get_scan(scan_name):
        match = sub_df[sub_df['scan_name'] == scan_name]
        if len(match) == 0:
            return None
        idx = match.index[0]
        return scans[idx]

    rest1 = get_scan('rest_run-1')
    img1  = get_scan('imagery_run-1')
    img2  = get_scan('imagery_run-2')
    img3  = get_scan('imagery_run-3')
    img4  = get_scan('imagery_run-4')
    rest2 = get_scan('rest_run-2')

    # Definer perioder
    periods = {}

    if rest1 is not None:
        periods['rest-1'] = rest1

    if img1 is not None and img2 is not None:
        periods['pre-LOR'] = np.concatenate([img1, img2[:lor_tr]])

    if pd.notna(ror_tr) and img2 is not None and img3 is not None:
        ror_tr_int = int(ror_tr)
        periods['LOR']      = np.concatenate([img2[lor_tr:], img3[:ror_tr_int]])
        if img4 is not None:
            periods['post-LOR'] = np.concatenate([img3[ror_tr_int:], img4])

    if rest2 is not None:
        periods['rest-2'] = rest2

    # Beregn FO for hver periode
    for period_name, period_data in periods.items():
        if len(period_data) > 0:
            for k in range(K):
                rows_periods.append({
                    'subject': sub,
                    'period':  period_name,
                    'state':   f'state_{k+1}',
                    'FO':      np.sum(period_data == k) / len(period_data)
                })

df_periods = pd.DataFrame(rows_periods)
print(df_periods.shape)

def permutation_test_maxT(df_periods, period_ref, period_test, K, n_permutations=10000):
    
    # Find subjekter med begge perioder
    subs_ref  = set(df_periods[df_periods['period'] == period_ref]['subject'].unique())
    subs_test = set(df_periods[df_periods['period'] == period_test]['subject'].unique())
    subs_both = subs_ref & subs_test
    print(f"Subjekter med begge perioder: {len(subs_both)}")
    
    df_filtered = df_periods[
        df_periods['period'].isin([period_ref, period_test]) &
        df_periods['subject'].isin(subs_both)
    ]

    t_observed = np.zeros(K)
    for k in range(K):
        df_state = df_filtered[df_filtered['state'] == f'state_{k+1}']
        x = df_state[df_state['period'] == period_ref]['FO'].values
        y = df_state[df_state['period'] == period_test]['FO'].values
        t_observed[k], _ = stats.ttest_rel(x, y)

    t_max_dist = np.zeros(n_permutations)
    for perm in range(n_permutations):
        t_perm = np.zeros(K)
        for k in range(K):
            df_state = df_filtered[df_filtered['state'] == f'state_{k+1}']
            x  = df_state[df_state['period'] == period_ref]['FO'].values
            y = df_state[df_state['period'] == period_test]['FO'].values
            signs = np.random.choice([-1, 1], size=len(x))
            diff_perm = signs * (y - x)
            t_perm[k], _ = stats.ttest_1samp(diff_perm, 0)
        t_max_dist[perm] = np.max(np.abs(t_perm))

    p_corrected = np.array([
        np.mean(t_max_dist >= np.abs(t_observed[k])) for k in range(K)
    ])

    return t_observed, p_corrected

# Sammenlign rest-1 vs LOR
print("\n rest-1 vs LOR:")
t, p = permutation_test_maxT(df_periods, 'rest-1', 'LOR', K)
for k in range(K):
    sig = '*' if p[k] < 0.05 else ''
    print(f"  State {k+1}: t={t[k]:.3f}, p={p[k]:.4f} {sig}")

    
# Sammenlign pre-LOR vs LOR
print("\npre-LOR vs LOR:")
t, p = permutation_test_maxT(df_periods, 'pre-LOR', 'LOR', K)
for k in range(K):
    sig = '*' if p[k] < 0.05 else ''
    print(f"  State {k+1}: t={t[k]:.3f}, p={p[k]:.4f} {sig}")


# Sammenlign post-LOR vs LOR
print("\n post-LOR vs LOR:")
t, p = permutation_test_maxT(df_periods, 'post-LOR', 'LOR', K)
for k in range(K):
    sig = '*' if p[k] < 0.05 else ''
    print(f"  State {k+1}: t={t[k]:.3f}, p={p[k]:.4f} {sig}")