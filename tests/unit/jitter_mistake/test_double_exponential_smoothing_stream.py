import numpy as np
import unittest

from jitter_smoothing.stream.implementations.double_exponentional_smoothing import (
    DoubleExponentialSmoothingStream,
)


class TestDoubleExponentialSmoothingStream(unittest.TestCase):
    def test_normal_case_return_ok(self):
        observation = np.array([[2, 3, 4]])
        expected_result = np.array([[1.44, 2.44, 3.44]])
        smoothing_stream = DoubleExponentialSmoothingStream(
            np.array([1, 2, 3]), alpha=0.4, beta=0.1
        )
        smoothing_stream.smooth_frame(observation)
        result = smoothing_stream.get_last_smoothed_frame()
        np.testing.assert_allclose(result, expected_result)

    def test_one_observation_returns_2D(self):
        observation = np.array([[2, 3]])
        expected_result = np.array([[1.44, 2.44]])
        smoothing_stream = DoubleExponentialSmoothingStream(
            np.array([1, 2]), alpha=0.4, beta=0.1
        )
        smoothing_stream.smooth_frame(observation)
        result = smoothing_stream.get_last_smoothed_frame()
        np.testing.assert_allclose(result, expected_result)

    def test_alpha_zero_returns_first(self):
        observation = np.array([[2, 3, 4]])
        expected_result = np.array([[1.0, 2.0, 3.0]])
        smoothing_stream = DoubleExponentialSmoothingStream(
            np.array([1, 2, 3]), alpha=0, beta=0.1
        )
        smoothing_stream.smooth_frame(observation)
        result = smoothing_stream.get_last_smoothed_frame()
        np.testing.assert_allclose(result, expected_result)

    def test_alpha_one_beta_one_return_plus_one(self):
        observation = np.array([[2, 3, 4]])
        expected_result = np.array([[3.0, 4.0, 5.0]])
        smoothing_stream = DoubleExponentialSmoothingStream(
            np.array([1, 2, 3]), alpha=1, beta=1
        )
        smoothing_stream.smooth_frame(observation)
        result = smoothing_stream.get_last_smoothed_frame()
        np.testing.assert_allclose(result, expected_result)

    def test_alpha_one_beta_0_1_return_plus_one(self):
        observation = np.array([[2, 3, 4]])
        expected_result = np.array([[2.1, 3.1, 4.1]])
        smoothing_stream = DoubleExponentialSmoothingStream(
            np.array([1, 2, 3]), alpha=1, beta=0.1
        )
        smoothing_stream.smooth_frame(observation)
        result = smoothing_stream.get_last_smoothed_frame()
        np.testing.assert_allclose(result, expected_result)
