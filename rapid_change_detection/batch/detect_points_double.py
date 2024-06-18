import numpy as np


class _DoubleExponentialSmoothingStream:
    def __init__(self, initial_frame, alpha=0.6, beta=0.1):
        self.current_state = initial_frame
        self.level_component = initial_frame
        self.trend_component = np.zeros_like(initial_frame)
        self.post_predict_current_state = np.zeros_like(initial_frame)
        self.post_predict_level_component = np.zeros_like(initial_frame)
        self.post_predict_trend_component = np.zeros_like(initial_frame)
        self.alpha = alpha
        self.beta = beta

    def predict_joint(self, frame):
        curr_cords = []
        new_level_component = np.zeros_like(self.current_state)
        new_trend_component = np.zeros_like(self.current_state)
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


def detect_points(data, max_num_to_compute=5, alpha=0.2, beta=0.1, threshold=0.75):
    num_of_frames = data.shape[0]
    num_of_joints = data.shape[1]
    data = np.transpose(data, (1, 0, 2))
    result = [set() for _ in range(num_of_joints)]
    for i, joint in enumerate(data):
        if not joint.any():
            continue
        num_of_detected = 0
        des = _DoubleExponentialSmoothingStream(joint[0], alpha, beta)
        for j, frame in enumerate(joint[1:], 2):
            des.predict_joint(frame)
            predicted = des.get_last_predicted_frame()

            if (abs(predicted - frame) > threshold).any() or np.isnan(np.sum(frame)):
                result[i].add(j)
                num_of_detected += 1
            else:
                des.update(frame)
                num_of_detected = 0
            if num_of_detected > max_num_to_compute:
                result[i].update(range(j, num_of_frames))
                break

    return [sorted(list(points)) for points in result]
