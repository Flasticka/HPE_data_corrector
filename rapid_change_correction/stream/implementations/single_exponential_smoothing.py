import numpy as np
from ..interfaces.rapid_change_repair_stream_interface import (
    RapidChangeRepairStreamInterface,
)


class SingleExponentialSmoothingStream:
    def __init__(self, initial_frame, alpha=0.2):
        self.current_state = np.array(initial_frame)
        self.predicted = np.array(initial_frame)
        self.alpha = alpha

    def predict_joint(self, frame):
        self.predicted = self.alpha * frame + (1 - self.alpha) * self.current_state

    def update(self, frame):
        self.current_state = np.array(frame)

    def get_last_predicted_frame(self):
        return self.predicted


class SingleExponentionalSmoothingRepair(RapidChangeRepairStreamInterface):
    def __init__(
        self,
        detection_component,
        number_of_joints,
        alpha=0.2,
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
            ses = SingleExponentialSmoothingStream(observation, self.alpha)
            ses.predict_joint(observation)
            self.state[joint_number] = ses
            return np.array(observation)
