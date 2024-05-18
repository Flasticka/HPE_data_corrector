import unittest
import numpy as np

from jitter_smoothing.batch.single_exponentional_smoothing import (
    _exponential_smoothing,
    smooth,
)


class TestExponentialSmoothing(unittest.TestCase):
    def test_normal_case_return_ok(self):
        observations = np.array([1, 2, 3, 4, 5])
        alpha = 0.4
        expected_result = np.array([1.0, 1.4, 2.04, 2.824, 3.6944])
        result = _exponential_smoothing(observations, alpha)
        np.testing.assert_allclose(result, expected_result)

    def test_one_observation_returns_ok(self):
        observations = np.array([1])
        alpha = 0.4
        expected_result = np.array([1.0])
        result = _exponential_smoothing(observations, alpha)
        np.testing.assert_allclose(result, expected_result)

    def test_alpha_zero_returns_ones(self):
        observations = np.array([1, 2, 3, 4, 5])
        alpha = 0
        expected_result = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = _exponential_smoothing(observations, alpha)
        np.testing.assert_allclose(result, expected_result)

    def test_alpha_one_return_same(self):
        observations = np.array([1, 2, 3, 4, 5])
        alpha = 1
        expected_result = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _exponential_smoothing(observations, alpha)
        np.testing.assert_allclose(result, expected_result)

    def test_no_observation_returns_ok(self):
        observations = np.array([])
        alpha = 0.4
        expected_result = np.array([])
        result = _exponential_smoothing(observations, alpha)
        np.testing.assert_allclose(result, expected_result)


class TestSmooth(unittest.TestCase):
    def test_smooth_normal_case_returns_ok(self):
        motion_data = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[2, 3, 4], [5, 6, 7]],
                [[3, 4, 5], [6, 7, 8]],
                [[4, 5, 6], [7, 8, 9]],
            ]
        )

        expected_result = np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[1.4, 2.4, 3.4], [4.4, 5.4, 6.4]],
                [[2.04, 3.04, 4.04], [5.04, 6.04, 7.04]],
                [[2.824, 3.824, 4.824], [5.824, 6.824, 7.824]],
            ]
        )

        result = smooth(motion_data, alpha=0.4)
        np.testing.assert_array_almost_equal(result, expected_result, decimal=3)

    def test_smooth_one_frame_return_ok(self):
        motion_data = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
            ]
        )

        expected_result = np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            ]
        )
        result = smooth(motion_data, alpha=0.4)
        np.testing.assert_array_almost_equal(result, expected_result, decimal=3)
