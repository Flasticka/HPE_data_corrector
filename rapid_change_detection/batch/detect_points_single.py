import numpy as np


class _SingleExponentialSmoothingStream:
    def __init__(self, initial_frame, alpha=0.6):
        self.current_state = initial_frame
        self.predicted = initial_frame
        self.alpha = alpha

    def predict_frame(self, frame):
        self.predicted = self.alpha * frame + (1 - self.alpha) * self.current_state

    def update(self, frame):
        self.current_state = frame

    def get_last_predict_frame(self):
        return self.predicted


def detect_points(data, max_num_to_compute=15, alpha=0.6, threshold=0.02):
    num_of_frames = data.shape[1]
    num_of_joints = data.shape[0]
    result = [set() for _ in range(num_of_joints)]
    predicted_ar = []
    for i, joint in enumerate(data):
        if not joint.any():
            continue
        num_of_detected = 0
        ses = _SingleExponentialSmoothingStream(joint[0], alpha)
        predicted_ar.append([joint[0].tolist()])

        for j, frame in enumerate(joint[1:], 2):
            ses.predict_frame(frame)
            predicted = ses.get_last_predict_frame()

            if (abs(predicted[:2] - frame[:2]) > threshold).any():  # or (
                # len(frame) == 3 and (abs(predicted[2] - frame[2]) > threshold)
                # ):
                if i == 16:
                    print(j, predicted, frame)
                predicted_ar[i].append(predicted.tolist())
                result[i].add(j)
                num_of_detected += 1
            else:
                predicted_ar[i].append(predicted.tolist())
                ses.update(frame)
                num_of_detected = 0

            if num_of_detected > max_num_to_compute:
                predicted_ar[i].append([])
                result[i].update(range(j, num_of_frames))
                break

    return [sorted(list(points)) for points in result], predicted_ar
