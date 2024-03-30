import numpy as np


def exponential_smoothing(observations, alpha, beta):
    result = np.zeros(observations.shape[0])
    result[0] = observations[0]
    level_component = 0
    trend_component = 0
    for t in range(1, observations.shape[0]):
        new_level_component = alpha * observations[t - 1] + (1 - alpha) * (level_component - trend_component)
        trend_component = beta * (new_level_component - level_component) + (1 - beta) * trend_component
        result[t] = new_level_component + trend_component
        level_component = new_level_component
    return result


def smooth(motion_data, data_dimension, smoothing_factor=0.4, trend_factor=0.4):
    result = []
    for joint in motion_data:
        single_smoothing_series = []
        for j in range(data_dimension):
            one_dimensional = joint[:, j]
            single_smoothing = exponential_smoothing(one_dimensional, smoothing_factor, trend_factor)
            single_smoothing_series.append(single_smoothing)
        joined_exponential_3D = np.array(single_smoothing_series).T.tolist()
        result.append(joined_exponential_3D)
    return result
