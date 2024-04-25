import numpy as np

from exceptions.not_valid_dimensionality_exception import (
    NotValidDimensionalityException,
)


def _exponential_smoothing(observations, alpha):
    result = np.zeros(observations.shape[0])
    if len(observations) > 0:
        result[0] = observations[0]
    for t in range(1, observations.shape[0]):
        result[t] = alpha * observations[t] + (1 - alpha) * result[t - 1]
    return result


def smooth(motion_data, alpha=0.4):
    result = []
    if len(motion_data.shape) != 3:
        raise NotValidDimensionalityException("motion_data", 3)
    motion_data = np.transpose(motion_data, (1, 0, 2))
    for joint in motion_data:
        single_smoothing_series = []
        for j in range(motion_data.shape[2]):
            one_dimensional = joint[:, j]
            single_smoothing = _exponential_smoothing(one_dimensional, alpha)
            single_smoothing_series.append(single_smoothing)
        joined_exponential_3D = np.array(single_smoothing_series).T.tolist()
        result.append(joined_exponential_3D)

    return np.transpose(result, (1, 0, 2))
