import numpy as np

from exceptions.not_valid_dimensionality_exception import (
    NotValidDimensionalityException,
)


def _exponential_smoothing(observations, alpha, beta):
    result = np.zeros(observations.shape[0])
    if len(observations) > 0:
        result[0] = observations[0]
    else:
        return result
    level, trend = result[0], 0

    for t in range(1, len(observations)):
        new_level = alpha * observations[t] + (1 - alpha) * (level - trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        result[t] = new_level + trend
        level = new_level
    return result


def smooth(motion_data, alpha=0.4, beta=0.1):
    result = []
    if len(motion_data.shape) != 3:
        raise NotValidDimensionalityException("motion_data", 3)
    motion_data = np.transpose(motion_data, (1, 0, 2))

    for joint in motion_data:
        double_smoothing_series = []
        for j in range(motion_data.shape[2]):
            one_dimensional = joint[:, j]
            double_smoothing = _exponential_smoothing(one_dimensional, alpha, beta)
            double_smoothing_series.append(double_smoothing)
        joined_exponential = np.array(double_smoothing_series).T.tolist()
        result.append(joined_exponential)

    return np.transpose(result, (1, 0, 2))
