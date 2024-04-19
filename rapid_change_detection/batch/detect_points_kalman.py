from filterpy.kalman import KalmanFilter
import numpy as np
import math


def _initiate_kalman(observation, q, r):
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


def _initiate_kalman2D(observation, q, r):
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


def detect_points(data, max_num_to_compute=15, q=1, r=1, threshold=0.02):
    num_of_joints, num_of_frames, dimension = data.shape
    result = [set() for _ in range(num_of_joints)]
    predicted = []

    for i, joint in enumerate(data):
        if len(joint) == 0:
            continue
        num_of_detected = 0
        kf = (
            _initiate_kalman2D(joint[0], q, r)
            if dimension == 2
            else _initiate_kalman2D(joint[0][:2], q, r)
        )
        predicted.append([joint[0].tolist()])
        for j, frame in enumerate(joint[1:], 2):
            kf.predict()

            predicted_dim = kf.x_prior[:2, 0] if dimension == 2 else kf.x_prior[:2, 0]

            frame_dim = frame[:2] if dimension == 2 else frame[:2]

            if (abs(predicted_dim - frame_dim) > threshold).any() and j > 5:
                if i == 16:
                    print(j, predicted_dim, frame_dim)
                predicted[i].append(predicted_dim.tolist())
                result[i].add(j)
                num_of_detected += 1
            else:

                kf.update(frame_dim)

                predicted[i].append(predicted_dim.tolist())
                num_of_detected = 0

            if num_of_detected > max_num_to_compute:
                result[i].update(range(j, num_of_frames))
                predicted[i].append([])

                break

    return [sorted(list(points)) for points in result], predicted
