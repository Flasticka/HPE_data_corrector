from pykalman import KalmanFilter
import numpy as np



def _smooth_kalman(joint, q, r):
    kf = KalmanFilter(
        transition_matrices=np.array(
            [
                [1, 0, 0, 1, 0, 0, 0.5, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0.5, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 0.5],
                [0, 0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        ),
        transition_covariance=q * np.eye(9),
        observation_covariance=r * np.eye(3),
        n_dim_state=9,
        n_dim_obs=3,
        initial_state_mean=[joint[0, 0], joint[0, 1], joint[0, 2], 0, 0, 0, 0, 0, 0],
    )

    return kf.smooth(joint[:, :3])[0][:, 0:3]


def _smooth_kalman2D(joint, q, r):
    kf = KalmanFilter(
        transition_matrices=np.array(
            [
                [1, 0, 1, 0, 0.5, 0],
                [0, 1, 0, 1, 0, 0.5],
                [0, 0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        ),
        transition_covariance=q * np.eye(6),
        observation_covariance=r * np.eye(2),
        n_dim_state=6,
        n_dim_obs=2,
        initial_state_mean=[joint[0, 0], joint[0, 1], 0, 0, 0, 0],
    )
    return kf.smooth(joint[:, :2])[0][:, 0:2]


def smooth(motion_data, q=0.05, r=1):
    result = []
    motion_data = np.transpose(motion_data, (1, 0, 2))

    if motion_data.shape[2] == 2:
        smoothing_function = _smooth_kalman2D
    else:
        smoothing_function = _smooth_kalman

    for joint in motion_data:
        result_current = []
        intervals = get_intervals(joint)
        for interval in intervals:
            kalman_smoothing = smoothing_function(interval, q, r)
            result_current += kalman_smoothing.tolist()
        result.append(result_current)
    return np.array(np.transpose(result, (1, 0, 2)))


def get_intervals(joint):
    is_number = True
    result = []
    current = []
    for element in joint:
        if np.isnan(element).any():
            if is_number and len(current) > 0:
                result.append(np.array(current))
                current = []
            current.append(element)
            is_number = False
        else:
            if not is_number and len(current) > 0:
                result.append(np.array(current))
                current = []
            current.append(element)
            is_number = True
    if len(current) > 0:
        result.append(np.array(current))
    return result
