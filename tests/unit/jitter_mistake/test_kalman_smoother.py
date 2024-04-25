import unittest
import numpy as np

from exceptions.not_valid_dimensionality_exception import (
    NotValidDimensionalityException,
)
from exceptions.not_valid_length_exception import NotValidLengthException
from jitter_smoothing.batch.kalman_smoother import (
    smooth,
)


class TestSmooth(unittest.TestCase):
    def test_smooth_normal_case_returns_ok_and_good_shape(self):
        motion_data = np.zeros((10, 5, 2))
        smoothed_data = smooth(motion_data)
        assert smoothed_data.shape == (10, 5, 2)

    def test_smooth_one_frame_raise_exception(self):
        motion_data = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
            ]
        )

        try:
            smooth(motion_data)
        except NotValidLengthException as err:
            self.assertEqual(
                "Invalid length of variable 'motion_data', expected to have at least 2 elements.",
                str(err),
            )

    def test_smooth_two_frame_return_ok(self):
        motion_data = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[2, 3, 4], [5, 6, 7]],
            ]
        )

        expected_result = np.array(
            [
                [
                    [1.1785, 2.1785, 3.1785],
                    [4.1785, 5.1785, 6.1785],
                ],
                [
                    [1.6428, 2.6428, 3.6428],
                    [4.6428, 5.6428, 6.6428],
                ],
            ]
        )
        result = smooth(motion_data)
        np.testing.assert_array_almost_equal(result, expected_result, decimal=3)

    def test_smooth_two_frame2D_return_ok(self):
        motion_data = np.array(
            [
                [[1, 2], [4, 5]],
                [[2, 3], [5, 6]],
            ]
        )

        expected_result = np.array(
            [
                [
                    [1.1785, 2.1785],
                    [4.1785, 5.1785],
                ],
                [
                    [1.6428, 2.6428],
                    [4.6428, 5.6428],
                ],
            ]
        )
        result = smooth(motion_data)
        np.testing.assert_array_almost_equal(result, expected_result, decimal=3)

    def test_smooth_frame1D_raise_exception(self):
        motion_data = np.array(
            [
                [[1], [4]],
                [[2], [5]],
            ]
        )
        try:
            smooth(motion_data)
        except NotValidDimensionalityException as err:
            self.assertEqual(
                "Invalid dimensionality of variable 'motion_data.shape[2] (cords dimensionality)', expected "
                "dimensionality 2 or 3.",
                str(err),
            )

    def test_smooth_no_frame_raise_exception(self):
        motion_data = np.array([])

        try:
            smooth(motion_data)
        except NotValidDimensionalityException as err:
            self.assertEqual(
                "Invalid dimensionality of variable 'motion_data', expected dimensionality 3.",
                str(err),
            )
