import unittest
import numpy as np
from jitter_smoothing.batch.kalman_smoother import (
    smooth,
)


class TestSmooth(unittest.TestCase):
    def test_smooth_normal_case_returns_ok_and_good_shape(self):
        motion_data = np.zeros((10, 5, 2))
        smoothed_data = smooth(motion_data)
        assert smoothed_data.shape == (10, 5, 2)

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
