from pykalman import KalmanFilter
import numpy as np

from exceptions.not_valid_dimensionality_exception import NotValidDimensionalityException
from exceptions.not_valid_length_exception import NotValidLengthException


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
    if len(motion_data.shape) != 3:
        raise NotValidDimensionalityException("motion_data", 3)
    if len(motion_data) < 2:
        raise NotValidLengthException("motion_data", "to have at least 2 elements")

    motion_data = np.transpose(motion_data, (1, 0, 2))

    if motion_data.shape[2] == 2:
        smoothing_function = _smooth_kalman2D
    elif motion_data.shape[2] == 3:
        smoothing_function = _smooth_kalman
    else:
        raise NotValidDimensionalityException("motion_data.shape[2] (cords dimensionality)", "2 or 3")

    for joint in motion_data:
        kalman_smoothing = smoothing_function(joint, q, r)
        result.append(kalman_smoothing.tolist())
    return np.array(np.transpose(result, (1, 0, 2)))
