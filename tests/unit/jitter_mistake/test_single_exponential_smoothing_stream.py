import numpy as np
import unittest

from jitter_smoothing.stream.implementations.single_exponentional_smoothing import (
    SingleExponentialSmoothingStream,
)


class TestSingleExponentialSmoothingStream(unittest.TestCase):
    def test_normal_case_return_ok(self):
        observation = np.array([[2, 3, 4]])
        expected_result = np.array([[1.4, 2.4, 3.4]])
        smoothing_stream = SingleExponentialSmoothingStream(
            np.array([1, 2, 3]), alpha=0.4
        )
        smoothing_stream.smooth_frame(observation)
        result = smoothing_stream.get_last_smoothed_frame()
        np.testing.assert_allclose(result, expected_result)

    def test_one_observation_returns_2D(self):
        observation = np.array([[2, 3]])
        expected_result = np.array([[1.4, 2.4]])
        smoothing_stream = SingleExponentialSmoothingStream(np.array([1, 2]), alpha=0.4)
        smoothing_stream.smooth_frame(observation)
        result = smoothing_stream.get_last_smoothed_frame()
        np.testing.assert_allclose(result, expected_result)

    def test_alpha_zero_returns_first(self):
        observation = np.array([[2, 3, 4]])
        expected_result = np.array([[1.0, 2.0, 3.0]])
        smoothing_stream = SingleExponentialSmoothingStream(
            np.array([1, 2, 3]), alpha=0
        )
        smoothing_stream.smooth_frame(observation)
        result = smoothing_stream.get_last_smoothed_frame()
        np.testing.assert_allclose(result, expected_result)

    def test_alpha_one_return_same(self):
        observation = np.array([[2, 3, 4]])
        expected_result = np.array([[2.0, 3.0, 4.0]])
        smoothing_stream = SingleExponentialSmoothingStream(
            np.array([1, 2, 3]), alpha=1
        )
        smoothing_stream.smooth_frame(observation)
        result = smoothing_stream.get_last_smoothed_frame()
        np.testing.assert_allclose(result, expected_result)
