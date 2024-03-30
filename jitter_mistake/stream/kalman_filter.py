from pykalman import KalmanFilter
import numpy as np


class KalmanFilterStream:
    def __init__(self, initial_frame, q=0.05):
        self.result = []
        self.kf_states = []
        self.kf_state_means = []
        self.kf_state_covariance = []
        for i in range(initial_frame.shape[0]):
            initial_state_mean = [
                initial_frame[i, 0],
                initial_frame[i, 1],
                initial_frame[i, 2],
                0,
                0,
                0,
                0,
                0,
                0,
            ]
            self.kf_states.append(
                KalmanFilter(
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
                    n_dim_state=9,
                    n_dim_obs=3,
                    initial_state_mean=initial_state_mean,
                )
            )
            self.kf_state_means.append(initial_state_mean)
            self.kf_state_covariance.append(q * np.eye(9))

    def smooth_frame(self, frame):
        self.result = []
        new_kf_state_means = []
        new_kf_state_covariance = []
        for i in range(frame.shape[0]):
            means, covariance = self.kf_states[i].filter_update(
                self.kf_state_means[i],
                self.kf_state_covariance[i],
                frame[i],
            )
            new_kf_state_means.append(means)
            new_kf_state_covariance.append(covariance)
            self.result.append(means[:3])
        self.kf_state_means = new_kf_state_means
        self.kf_state_covariance = new_kf_state_covariance

    def get_last_smoothed_frame(self):
        return np.array(self.result)


class KalmanFilter2DStream:
    def __init__(self, initial_frame, q=0.05):
        self.result = []
        self.kf_states = []
        self.kf_state_means = []
        self.kf_state_covariance = []
        for i in range(initial_frame.shape[0]):
            initial_state_mean = [initial_frame[i, 0], initial_frame[i, 1], 0, 0, 0, 0]
            self.kf_states.append(
                KalmanFilter(
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
                    n_dim_state=6,
                    n_dim_obs=3,
                    initial_state_mean=initial_state_mean,
                )
            )
            self.kf_state_means.append(initial_state_mean)
            self.kf_state_covariance.append(q * np.eye(6))

    def smooth_frame(self, frame):
        self.result = []
        new_kf_state_means = []
        new_kf_state_covariance = []
        for i in range(frame.shape[0]):
            means, covariance = self.kf_states[i].filter_update(
                self.kf_state_means[i],
                self.kf_state_covariance[i],
                frame[i],
            )
            new_kf_state_means.append(means)
            new_kf_state_covariance.append(covariance)
            self.result.append(means[:2])
        self.kf_state_means = new_kf_state_means
        self.kf_state_covariance = new_kf_state_covariance

    def get_last_smoothed_frame(self):
        return np.array(self.result)
