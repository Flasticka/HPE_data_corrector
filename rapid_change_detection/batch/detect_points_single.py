import numpy as np


class _SingleExponentialSmoothingStream:
    def __init__(self, initial_frame, alpha=0.2):
        self.current_state = initial_frame
        self.predicted = initial_frame
        self.alpha = alpha

    def predict_frame(self, frame):
        self.predicted = self.alpha * frame + (1 - self.alpha) * self.current_state

    def update(self, frame):
        self.current_state = frame

    def get_last_predict_frame(self):
        return self.predicted


def detect_points(data, max_num_to_compute=5, alpha=0.2, threshold=0.75):
    num_of_frames = data.shape[0]
    num_of_joints = data.shape[1]
    data = np.transpose(data, (1, 0, 2))
    result = [set() for _ in range(num_of_joints)]
    for i, joint in enumerate(data):
        if not joint.any():
            continue
        num_of_detected = 0
        ses = _SingleExponentialSmoothingStream(joint[0], alpha)

        for j, frame in enumerate(joint[1:], 2):
            ses.predict_frame(frame)
            predicted = ses.get_last_predict_frame()
            if (abs(predicted - frame) > threshold).any() or np.isnan(np.sum(frame)):
                result[i].add(j)
                num_of_detected += 1

            else:
                ses.update(frame)
                num_of_detected = 0
            if num_of_detected > max_num_to_compute:
                result[i].update(range(j, num_of_frames))
                break
    return [sorted(list(points)) for points in result]
