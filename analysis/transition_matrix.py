import numpy as np

# Extract X_part
X_part = np.loadtxt(r"C:\Users\Siri_\NRU\fMRI_internship\Analyse\diametrical_clustering_output\X_part.txt")
print(X_part.shape)

# Number of subjects in the study
n_subjects = 25 # sub-29 removed

# Scans lengths for each of the 6 scans per subject
scan_lengths_per_subject = [1145, 2270, 2270, 1145, 770, 770]

# Scan names for the 6 scans
scan_names= ["imagery-run-1", "imagery-run-2", "imagery-run-3", "imagery-run-4", "rest-run-1", "rest-run-2"]

all_lengths = scan_lengths_per_subject * n_subjects

# Partition X_part along time axis into separate scans
split_points = np.cumsum(all_lengths)[:-1]
scans = np.split(X_part, split_points)  

print(f"Number of scans: {len(scans)}")       # 150
print(f"First scan: {scans[0].shape}")    # (1145,)

# TRANSITION MATRIX
def _transition_matrix(labels, K):
    """
    Compute K×K Markov transition probability matrix from a state sequence.

    Parameters
    ----------
    labels : array-like, shape (T,)
        Integer state labels 0..K-1.
    K : int

    Returns
    -------
    P : np.ndarray, shape (K, K)
    """
    P = np.zeros((K, K))
    for t in range(len(labels) - 1):
        P[labels[t], labels[t + 1]] += 1

    # Normalize rows
    row_sums = P.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        P = np.where(row_sums > 0, P / row_sums, 0.0)

    return P

# Create transition matrix for all scans for all subjects
K=3

def get_matrix(scans, scan_names, K, n_subjects):
    """Calculate transition matrix for each subject and task.
    Returns array of shape (n_subjects, n_tasks, K, K)"""
    all_matrices = np.zeros((n_subjects, len(scan_names), K, K))

    for i in range(n_subjects):
        for t, task in enumerate(scan_names):
            actual_idx = i * len(scan_names) + t
            labels = scans[actual_idx].astype(int)
            all_matrices[i, t] = _transition_matrix(labels, K)

    return all_matrices  # (n_subjects, n_tasks, K, K)

matrices = get_matrix(scans, scan_names, K, n_subjects)
print(matrices)
