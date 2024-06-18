import math
import numpy as np
from ..interfaces.rapid_change_repair_stream_interface import (
    RapidChangeRepairStreamInterface,
)


class _DoubleExponentialSmoothingStream:

    def __init__(self, initial_frame, alpha=0.9, beta=0.1):
        self.current_state = initial_frame
        self.level_component = initial_frame
        self.trend_component = np.zeros(initial_frame.shape)
        self.post_predict_current_state = np.zeros(len(self.current_state))
        self.post_predict_level_component = np.zeros(len(self.level_component))
        self.post_predict_trend_component = np.zeros(len(self.trend_component))
        self.alpha = alpha
        self.beta = beta

    def predict_joint(self, frame):
        curr_cords = []
        new_level_component = np.zeros(len(self.current_state))
        new_trend_component = np.zeros(len(self.current_state))
        for j in range(len(self.current_state)):
            curr_level_component = self.alpha * frame[j] + (1 - self.alpha) * (
                self.level_component[j] + self.trend_component[j]
            )
            curr_trend_component = (
                self.beta * (curr_level_component - self.level_component[j])
                + (1 - self.beta) * self.trend_component[j]
            )
            curr_cords.append(curr_level_component + curr_trend_component)
            new_level_component[j] = curr_level_component
            new_trend_component[j] = curr_trend_component
        self.post_predict_current_state = curr_cords
        self.post_predict_level_component = new_level_component
        self.post_predict_trend_component = new_trend_component

    def update(self, frame):
        self.level_component = frame
        self.trend_component = self.post_predict_trend_component

    def get_last_predicted_frame(self):
        return np.array(self.post_predict_current_state)


class DoubleExponentialSmoothingRepair(RapidChangeRepairStreamInterface):
    def __init__(
        self,
        detection_component,
        number_of_joints,
        alpha=0.2,
        beta=0.1,
        max_frames_to_compute=5,
        previous_when_missing=False,
    ):
        self.number_of_joints = number_of_joints
        self.num_of_computed = [0] * number_of_joints
        self.is_computing = [False] * number_of_joints
        self.state = [None] * number_of_joints
        self.max_frames_to_compute = max_frames_to_compute
        self.previous_when_missing = previous_when_missing
        self.alpha = alpha
        self.beta = beta
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
                        result.append([None for _ in range(len(frame.shape[1]))])
            else:
                self.num_of_computed[joint_num] = 0
                self.update_stats(observation, joint_num, True)
                result.append(observation.tolist())
        return result

    def update_stats(self, observation, joint_number, update):
        if self.is_computing[joint_number]:
            self.state[joint_number].predict_joint(observation)
            predicted = self.state[joint_number].get_last_predicted_frame()
            if update:
                self.state[joint_number].update(observation)
            return predicted
        else:
            self.is_computing[joint_number] = True
            des = _DoubleExponentialSmoothingStream(observation, self.alpha, self.beta)
            des.predict_joint(observation)
            self.state[joint_number] = des
            return np.array(observation)
