import numpy as np
from ..interfaces.rapid_change_detection_interface import RapidChangeDetectionInterface


class DetectPointsBySingle(RapidChangeDetectionInterface):
    def __init__(self, initial_frame, alpha=0.6, threshold=0.4, max_num_to_compute=5):
        self.threshold = threshold
        self.max_num_to_compute = max_num_to_compute
        self.num_of_detected = [0] * initial_frame.shape[0]
        self.current_state = initial_frame
        self.previous_frame = initial_frame
        self.alpha = alpha

    def check_frame(self, frame):
        result = set()
        for i in range(self.current_state.shape[0]):
            if self.num_of_detected[i] > self.max_num_to_compute:
                result.add(i)
                continue
            predicted = [
                self.alpha * frame[i][j] + (1 - self.alpha) * self.current_state[i][j]
                for j in range(self.current_state.shape[1])
            ]

            if (
                    abs(np.array(predicted)[:2] - np.array(frame[i])[:2]) > self.threshold
            ).any() or (
                    len(frame[i]) == 3 and abs(predicted[2] - frame[i][2]) > self.threshold
            ):
                result.add(i)
                self.num_of_detected[i] += 1
            else:
                self.num_of_detected[i] = 0
                self.current_state[i] = frame[i]

        return result
