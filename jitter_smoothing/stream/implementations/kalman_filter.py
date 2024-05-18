from filterpy.kalman import KalmanFilter
import numpy as np

from ..interfaces.jitter_smoothing_interface import JitterSmoothingInterface


class KalmanFilterStream(JitterSmoothingInterface):
    def __init__(self, initial_frame, q=0.05, r=1):
        self.q = q
        self.r = r
        self.result = initial_frame
        self.kf_states = []
        for i in range(initial_frame.shape[0]):
            f = self._init(initial_frame[i], q, r)
            self.kf_states.append(f)

    def _init(self, frame, q, r):
        f = KalmanFilter(dim_x=9, dim_z=3)
        initial_state_mean = np.hstack((frame, np.zeros(6))).reshape(-1, 1)
        f.x = initial_state_mean
        f.F = np.array(
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
        )
        f.H = np.eye(3, 9)
        f.Q = q * np.eye(9)
        f.R = r * np.eye(3)
        return f

    def smooth_frame(self, frame):
        self.result = []
        for i in range(frame.shape[0]):
            if np.isnan(frame[i]).any():
                self.result.append(frame[i])
                self.kf_states[i] = None
            else:
                if self.kf_states[i] is not None:
                    self.kf_states[i].predict()
                    self.kf_states[i].update(frame[i])
                    self.result.append(self.kf_states[i].x_post[:3, 0])
                else:
                    self.kf_states[i] = self._init(frame[i], self.q, self.r)
                    self.result.append(frame[i])

    def get_last_smoothed_frame(self):
        return np.array(self.result)


class KalmanFilterStream2D:
    def __init__(self, initial_frame, q=0.05, r=1):
        self.q = q
        self.r = r
        self.result = initial_frame
        self.kf_states = []
        for i in range(initial_frame.shape[0]):
            f = self._init(initial_frame[i], q, r)
            self.kf_states.append(f)

    def _init(self, frame, q, r):
        f = KalmanFilter(dim_x=6, dim_z=2)
        initial_state_mean = np.hstack((frame, np.zeros(4))).reshape(-1, 1)
        f.x = initial_state_mean
        f.F = np.array(
            [
                [1, 0, 1, 0, 0.5, 0],
                [0, 1, 0, 1, 0, 0.5],
                [0, 0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        f.H = np.eye(2, 6)
        f.Q = q * np.eye(6)
        f.R = r * np.eye(2)
        return f

    def smooth_frame(self, frame):
        self.result = []
        for i in range(frame.shape[0]):
            if np.isnan(frame[i]).any():
                self.result.append(frame[i])
                self.kf_states[i] = None
            else:
                if self.kf_states[i] is not None:
                    self.kf_states[i].predict()
                    self.kf_states[i].update(frame[i])
                    self.result.append(self.kf_states[i].x_post[:2, 0])
                else:
                    self.kf_states[i] = self._init(frame[i], self.q, self.r)
                    self.result.append(frame[i])

    def get_last_smoothed_frame(self):
        return np.array(self.result)


def get_kalman(initial_frame, q=0.05, r=1):
    if initial_frame.shape[1] == 2:
        return KalmanFilterStream2D(initial_frame, q, r)
    else:
        return KalmanFilterStream(initial_frame, q, r)
