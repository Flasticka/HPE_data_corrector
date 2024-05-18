import numpy as np
from filterpy.kalman import KalmanFilter
from ..interfaces.rapid_change_repair_stream_interface import (
    RapidChangeRepairStreamInterface,
)


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


class KalmanFilterRepair(RapidChangeRepairStreamInterface):
    def __init__(
        self,
        detection_component,
        number_of_joints,
        q=0.01,
        r=1,
        max_frames_to_compute=5,
        previous_when_missing=False,
    ):
        self.number_of_joints = number_of_joints
        self.num_of_computed = [0] * number_of_joints
        self.is_computing = [False] * number_of_joints
        self.state = [None] * number_of_joints
        self.max_frames_to_compute = max_frames_to_compute
        self.previous_when_missing = previous_when_missing
        self.q = q
        self.r = r
        self.last_ok = [None] * number_of_joints
        super().__init__(detection_component)

    def repair_frame(self, frame):
        wrong_joints = self.detection_component.check_frame(frame)
        result = []
        for joint_num, observation in enumerate(frame):
            if joint_num in wrong_joints:
                if self.num_of_computed[joint_num] < self.max_frames_to_compute:
                    self.num_of_computed[joint_num] += 1
                    updated = self.update_stats(observation, joint_num, False)
                    result.append(updated.tolist())
                else:
                    self.is_computing[joint_num] = False
                    if self.previous_when_missing:
                        result.append(observation.tolist())
                    else:
                        result.append([None for _ in range(frame.shape[1])])
            else:
                self.last_ok[joint_num] = observation
                self.num_of_computed[joint_num] = 0
                self.update_stats(observation, joint_num, True)
                result.append(observation.tolist())
        return result

    def update_stats(self, observation, joint_number, update):
        if self.is_computing[joint_number]:
            self.state[joint_number].predict()
            predicted = (
                self.state[joint_number].x_prior[0:3, 0]
                if len(observation) == 3
                else self.state[joint_number].x_prior[0:2, 0]
            )
            if update:
                self.state[joint_number].update(observation)
            return predicted
        else:
            self.is_computing[joint_number] = True
            kalman = (
                initiate_kalman(observation, self.q, self.r)
                if len(observation) == 3
                else initiate_kalman2D(observation, self.q, self.r)
            )
            self.state[joint_number] = kalman
            return np.array(observation)
