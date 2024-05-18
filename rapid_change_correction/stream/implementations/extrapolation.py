import numpy as np
import scipy.interpolate as si
from ..interfaces.rapid_change_repair_stream_interface import (
    RapidChangeRepairStreamInterface,
)


class Extrapolation:
    def __init__(
        self, number_of_joints, predicted_points, dimension, look_back=10
    ) -> None:
        self.state = [[None for _ in range(3)] for _ in range(number_of_joints)]
        self.look_back = look_back
        self.predicted_points = predicted_points
        self.last_wrong = [0 for _ in range(number_of_joints)]
        self.dimension = dimension

    def initiate_extrapolation(self, joint, frame):

        for j in range(self.dimension):
            a = max(self.last_wrong[joint] + 1, frame - self.look_back)
            b = frame

            x = list(
                map(
                    lambda x: x[joint][j],
                    self.predicted_points[a:b],
                )
            )
            y = [
                i
                for i in range(
                    a,
                    b,
                )
            ]

            if len(x) < 2:
                return None
            extrapolate_func = si.interp1d(y, x, fill_value="extrapolate")

            self.state[joint][j] = extrapolate_func

    def compute_next(self, joint, frame):
        result = []
        for j in range(self.dimension):
            if self.state[joint][j] is None:
                return None
            result.append(self.state[joint][j](frame))
        return np.array(result)


class ExtrapolationRepair(RapidChangeRepairStreamInterface):
    def __init__(
        self,
        detection_component,
        number_of_joints,
        dimension=3,
        look_back=10,
        max_frames_to_compute=5,
        previous_when_missing=False,
    ):
        self.number_of_joints = number_of_joints
        self.num_of_computed = [0] * number_of_joints
        self.state = [None] * number_of_joints
        self.max_frames_to_compute = max_frames_to_compute
        self.previous_when_missing = previous_when_missing
        self.predicted_points = []
        self.extrapolation = Extrapolation(
            number_of_joints, self.predicted_points, dimension, look_back
        )
        self.frame_count = 0
        super().__init__(detection_component)

    def repair_frame(self, frame):
        wrong_joints = self.detection_component.check_frame(frame)
        result = []
        for joint_num, observation in enumerate(frame):
            if joint_num in wrong_joints:
                if self.num_of_computed[joint_num] < self.max_frames_to_compute:
                    if self.num_of_computed[joint_num] == 0:
                        self.extrapolation.initiate_extrapolation(
                            joint_num, self.frame_count
                        )
                    self.num_of_computed[joint_num] += 1
                    interpolated = self.extrapolation.compute_next(
                        joint_num, self.frame_count
                    )
                    if interpolated is None:
                        if self.previous_when_missing:
                            interpolated = np.array(frame[joint_num])
                        else:
                            interpolated = np.array(
                                [None for _ in range(frame.shape[1])]
                            )
                    result.append(interpolated.tolist())

                else:
                    self.extrapolation.last_wrong[joint_num] = self.frame_count

                    if self.previous_when_missing:
                        result.append(observation.tolist())
                    else:
                        result.append([None for _ in range(frame.shape[1])])
            else:
                self.num_of_computed[joint_num] = 0
                result.append(observation.tolist())
        self.frame_count += 1
        self.predicted_points.append(result[:])
        return result
