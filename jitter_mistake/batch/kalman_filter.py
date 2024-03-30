from pykalman import KalmanFilter
import numpy as np


def smooth_kalman(joint, covariance_coefficient):
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
        transition_covariance=covariance_coefficient * np.eye(9),
        n_dim_state=9,
        n_dim_obs=3,
        initial_state_mean=[
            joint[0, 0],
            joint[0, 1],
            joint[0, 2],
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    return kf.smooth(joint[:, :3])[0][:, 0:3]


def smooth_kalman2D(joint, covariance_coefficient):
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
        transition_covariance=covariance_coefficient * np.eye(6),
        n_dim_state=6,
        n_dim_obs=3,
        initial_state_mean=[
            joint[0, 0],
            joint[0, 1],
            0,
            0,
            0,
            0,
        ],
    )
    return kf.smooth(joint[:, :2])[0][:, 0:2]


def smooth(motion_data, covariance_coefficient=0.05):
    result = []
    for joint in motion_data:
        kalman_smoothing = smooth_kalman(joint, covariance_coefficient)
        result.append(kalman_smoothing.tolist())
    return result


def smooth2D(motion_data, covariance_coefficient=0.05):
    result = []
    for joint in motion_data:
        kalman_smoothing = smooth_kalman2D(joint, covariance_coefficient)
        result.append(kalman_smoothing.tolist())
    return result
