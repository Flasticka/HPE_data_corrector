import numpy as np


def combined_position_error(X, H, y):
    """
    Calculates the median of combined position error.

    Args:
        X: Collection of estimated state vectors.
        H: Measurement model matrix.
        y: Collection of measurements.

    Returns:
        The median of combined position error.
    """
    Dc = []  # List to store combined difference vectors
    for x_est in X:
        y_pred = H @ x_est  # Predicted measurement
        Di = y - y_pred  # Position difference vector
        Dc.append(np.linalg.norm(Di))  # Combined difference
    return np.median(Dc)


def one_hop_deviation(X, H):
    """
    Calculates the standard deviation of one hop position differences.

    Args:
        X: Collection of estimated state vectors.
        H: Measurement model matrix.

    Returns:
        The standard deviation of one hop position differences.
    """
    M = []  # List to store difference vectors
    for i in range(len(X) - 2):
        y_pred_i_plus_1 = H @ X[i + 1]
        y_pred_i = H @ X[i]
        Di = y_pred_i_plus_1 - y_pred_i
        Dc = np.linalg.norm(Di)
        M.append(Dc)
    return np.std(M)


def filter_response(X, y):
    """
    Calculates the standard deviation of filter response to measurement changes.

    Args:
        X: Collection of estimated state vectors.
        y: Collection of measurements.

    Returns:
        The standard deviation of filter response to measurement changes.
    """
    DX = np.diff(y, axis=0)  # Differences between adjacent measurement elements
    DY = np.diff(X, axis=0)  # Differences between adjacent estimated state vectors

    DXC = np.linalg.norm(DX, axis=1)  # Combined differences for measurements
    DYC = np.linalg.norm(DY, axis=1)  # Combined differences for estimates

    # Group n elements for calculating standard deviation (adjust n as needed)
    n = 5
    MXC = np.convolve(DXC, np.ones(n), mode="valid") / n
    MYC = np.convolve(DYC, np.ones(n), mode="valid") / n

    K = MXC - MYC
    return np.std(K)


def combine_metrics(X, H, y):
    """
    Combines metrics to find optimal Q and R values using a heuristic approach.

    Args:
        P1: Median of combined position error (scaled).
        P2: Standard deviation of one hop position differences (scaled).
        P3: Standard deviation of filter response (scaled).

    Returns:
        Estimated optimal values of Q and R (heuristic approach).
    """
    # Scaling factors (adjust if necessary)
    scale_p1 = 2.75
    scale_p2 = 2.75

    P1 = combined_position_error(X, H, y)
    P2 = one_hop_deviation(X, H)
    P3 = filter_response(X, y)
    Cr1 = P1 / scale_p1  # Mean of rows in P1 (scaled)
    Cr3 = P3  # Mean of rows in P3

    DF1 = np.max(Cr1) - np.min(Cr1)  # Difference between max and min in Cr1
    # Heuristic to estimate shift for Cr3: adjust polynomial degree or function
    shift = 0.1 * DF1**2  # Replace with a more suitable polynomial fit

    Cr3_adjusted = 2 * np.mean(Cr3) - Cr3 + shift  # Recalculate Cr3 with shift

    Q = np.min(np.maximum(Cr1, Cr3_adjusted), axis=0)  # Find Q

    Cc2 = P2 / scale_p2  # Mean of columns in P2 (scaled)
    Cc3 = P3  # Mean of columns in P3

    DF2 = np.max(Cc2) - np.min(Cc2)  # Difference between max and min in Cc2
    # Heuristic to estimate shift for Cc3: adjust polynomial degree or function
    shift = 0.1 * DF2**2  # Replace with a more suitable polynomial fit

    Cc3_adjusted = Cc3 + shift  # Recalculate Cc3 with shift

    R = np.min(np.maximum(Cc2, Cc3_adjusted))  # Find R

    return Q, R


def optimalize_for_joint(X, H, y):
    result = []
    X = np.transpose(X, (1, 0, 2))
    y = np.transpose(y, (1, 0, 2))
    for i, x_joint in enumerate(X):
        result.append(combine_metrics(x_joint, H, y[i]))
    return result
