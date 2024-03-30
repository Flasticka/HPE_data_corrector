import numpy as np


def exponential_smoothing(observations, alpha):
    result = np.zeros(observations.shape[0])
    result[0] = observations[0]
    for t in range(1, observations.shape[0]):
        result[t] = alpha * observations[t - 1] + (1 - alpha) * result[t - 1]
    return result


def smooth(motion_data, data_dimension, smoothing_factor=0.4):
    result = []
    for joint in motion_data:
        single_smoothing_series = []
        for j in range(data_dimension):
            one_dimensional = joint[:, j]
            single_smoothing = exponential_smoothing(one_dimensional, smoothing_factor)
            single_smoothing_series.append(single_smoothing)
        joined_exponential_3D = np.array(single_smoothing_series).T.tolist()
        result.append(joined_exponential_3D)
    return result
