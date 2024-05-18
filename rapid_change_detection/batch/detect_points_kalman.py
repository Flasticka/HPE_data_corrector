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


def detect_points(data, max_num_to_compute=5, q=0.05, r=1, threshold=0.75):
    num_of_frames, num_of_joints, dimension = data.shape
    result = [set() for _ in range(num_of_joints)]
    data = np.transpose(data, (1, 0, 2))
    for i, joint in enumerate(data):
        if len(joint) == 0:
            continue
        num_of_detected = 0
        kf = (
            _initiate_kalman(joint[0], q, r)
            if dimension == 3
            else _initiate_kalman2D(joint[0][:2], q, r)
        )
        for j, frame in enumerate(joint[1:], 2):
            kf.predict()

            predicted_dim = kf.x_prior[:2, 0] if dimension == 2 else kf.x_prior[:3, 0]
            frame_dim = frame[:2] if dimension == 2 else frame
            if (abs(predicted_dim - frame_dim) > threshold).any() or np.isnan(
                np.sum(frame_dim)
            ):
                kf.update(predicted_dim)
                result[i].add(j)
                num_of_detected += 1
            else:
                kf.update(frame_dim)
                num_of_detected = 0
            if num_of_detected > max_num_to_compute:
                result[i].update(range(j, num_of_frames))
                break
    return [sorted(list(points)) for points in result]
