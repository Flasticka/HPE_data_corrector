from filterpy.kalman import KalmanFilter
import numpy as np
from ..interfaces.rapid_change_detection_interface import RapidChangeDetectionInterface


def initiate_kalman(observation, q, r):
    f = KalmanFilter(dim_x=9, dim_z=3)
    initial_state_mean = np.hstack((observation, np.zeros(6))).reshape(-1, 1)
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


def initiate_kalman2D(observation, q, r):
    f = KalmanFilter(dim_x=6, dim_z=2)
    initial_state_mean = np.hstack((observation, np.zeros(4))).reshape(-1, 1)
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


class DetectPointsByKalman(RapidChangeDetectionInterface):
    def __init__(
            self,
            initial_frame,
            q=0.05,
            r=0.05,
            threshold=0.4,
            max_num_to_compute=5,
    ) -> None:
        self.threshold = threshold
        self.max_num_to_compute = max_num_to_compute
        self.num_of_detected = [0 for _ in range(initial_frame.shape[0])]
        self.kf_states = []
        self.dimension = initial_frame.shape[1]
        for i in range(initial_frame.shape[0]):
            f = None
            if self.dimension == 2:
                f = initiate_kalman2D(initial_frame[i], q, r)
            if self.dimension == 3:
                f = initiate_kalman(initial_frame[i], q, r)
            self.kf_states.append(f)

    def check_frame(self, frame):
        result = set()
        for i in range(frame.shape[0]):
            if self.num_of_detected[i] > self.max_num_to_compute:
                result.add(i)
                continue
            self.kf_states[i].predict()
            predicted_dim = (
                self.kf_states[i].x_prior[:2, 0]
                if self.dimension == 2
                else self.kf_states[i].x_prior[:3, 0]
            )
            frame_dim = frame[i][:2] if self.dimension == 2 else frame[i]

            if (abs(predicted_dim - frame_dim) > self.threshold).any() or (
                    len(frame[i]) == 3
                    and abs(predicted_dim[2] - frame[i][2]) > self.threshold
            ):
                result.add(i)
                self.num_of_detected[i] += 1
            else:
                self.kf_states[i].update(frame[i])
                self.num_of_detected[i] = 0

        return result
