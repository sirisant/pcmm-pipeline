import numpy as np

def global_signal_regression(data):
    """
    Perform global signal regression (GSR).

    Parameters
    ----------
    data : ndarray, shape (time, space)
        Time × space data matrix.

    Returns
    -------
    data_gsr : ndarray, shape (time, space)
        Data after regressing out the global signal.
    """
    gs = data.mean(axis=1, keepdims=True)  # (time, 1)

    # Regression coefficient for each spatial location
    beta = (data.T @ gs) / (gs.T @ gs)     # (space, 1)

    # Predicted global-signal contribution
    fitted = gs @ beta.T                   # (time, space)

    return data - fitted