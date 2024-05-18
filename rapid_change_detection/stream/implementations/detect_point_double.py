import numpy as np
from ..interfaces.rapid_change_detection_interface import RapidChangeDetectionInterface


class DetectPointsByDouble(RapidChangeDetectionInterface):
    def __init__(
        self, initial_frame, alpha=0.2, beta=0.1, threshold=0.75, max_num_to_compute=5
    ) -> None:
        self.threshold = threshold
        self.max_num_to_compute = max_num_to_compute
        self.num_of_detected = [0 for _ in range(initial_frame.shape[0])]
        self.alpha = alpha
        self.beta = beta
        self.level_component = initial_frame
        self.trend_component = np.zeros(initial_frame.shape)

    def check_frame(self, frame):
        result = set()
        new_state = np.zeros(self.level_component.shape)
        new_level_component = np.zeros(self.level_component.shape)
        new_trend_component = np.zeros(self.level_component.shape)
        for i in range(self.level_component.shape[0]):
            if self.num_of_detected[i] > self.max_num_to_compute:
                result.add(i)
                continue
            curr_joint_level_component = []
            curr_joint_trend_component = []
            for j in range(self.level_component.shape[1]):
                curr_level_component = self.alpha * frame[i][j] + (1 - self.alpha) * (
                    self.level_component[i][j] - self.trend_component[i][j]
                )
                curr_trend_component = (
                    self.beta * (curr_level_component - self.level_component[i][j])
                    + (1 - self.beta) * self.trend_component[i][j]
                )
                new_state[i][j] = curr_level_component + curr_trend_component
                curr_joint_level_component.append(curr_level_component)
                curr_joint_trend_component.append(curr_trend_component)
            if (abs(new_state[i] - frame[i]) > self.threshold).any() or np.isnan(
                np.sum(frame[i])
            ):
                result.add(i)
                self.num_of_detected[i] += 1
                new_level_component[i] = np.array(self.level_component[i])
                new_trend_component[i] = np.array(self.trend_component[i])
            else:
                new_level_component[i] = np.array(frame[i])
                new_trend_component[i] = np.array(curr_joint_trend_component)
                self.num_of_detected[i] = 0
        self.level_component = new_level_component
        self.trend_component = new_trend_component
        return result
