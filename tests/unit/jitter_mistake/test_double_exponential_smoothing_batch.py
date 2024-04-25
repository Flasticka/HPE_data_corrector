import unittest
import numpy as np

from exceptions.not_valid_dimensionality_exception import (
    NotValidDimensionalityException,
)
from jitter_smoothing.batch.double_exponentional_smoothing import (
    _exponential_smoothing,
    smooth,
)


class TestExponentialSmoothing(unittest.TestCase):
    def test_normal_case_return_ok(self):
        observations = np.array([1, 2, 3, 4, 5])
        alpha = 0.4
        beta = 0.1
        expected_result = np.array([1.0, 1.44, 2.1136, 2.912384, 3.779305])
        result = _exponential_smoothing(observations, alpha, beta)
        np.testing.assert_allclose(result, expected_result)

    def test_one_observation_returns_ok(self):
        observations = np.array([1])
        alpha = 0.4
        beta = 0.1

        expected_result = np.array([1.0])
        result = _exponential_smoothing(observations, alpha, beta)
        np.testing.assert_allclose(result, expected_result)

    def test_alpha_zero_returns_ones(self):
        observations = np.array([1, 2, 3, 4, 5])
        alpha = 0
        beta = 0.1
        expected_result = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = _exponential_smoothing(observations, alpha, beta)
        np.testing.assert_allclose(result, expected_result)

    def test_alpha_one_beta_one_return_plus_one(self):
        observations = np.array([1, 2, 3, 4, 5])
        alpha = 1
        beta = 1

        expected_result = np.array([1.0, 3.0, 4.0, 5.0, 6.0])
        result = _exponential_smoothing(observations, alpha, beta)
        np.testing.assert_allclose(result, expected_result)

    def test_no_observation_returns_ok(self):
        observations = np.array([])
        alpha = 0.4
        beta = 0.1

        expected_result = np.array([])
        result = _exponential_smoothing(observations, alpha, beta)
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
                [[1.44, 2.44, 3.44], [4.44, 5.44, 6.44]],
                [[2.1136, 3.1136, 4.1136], [5.1136, 6.1136, 7.1136]],
                [
                    [2.912384, 3.912384, 4.9123839],
                    [5.9123839, 6.9123839, 7.9123839],
                ],
            ]
        )

        result = smooth(motion_data, alpha=0.4, beta=0.1)
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
        result = smooth(motion_data, alpha=0.4, beta=0.1)
        np.testing.assert_array_almost_equal(result, expected_result, decimal=3)

    def test_smooth_no_frame_raise_exception(self):
        motion_data = np.array([])

        try:
            smooth(motion_data, alpha=0.4, beta=0.1)
        except NotValidDimensionalityException as err:
            self.assertEqual(
                "Invalid dimensionality of variable 'motion_data', expected dimensionality 3.",
                str(err),
            )
