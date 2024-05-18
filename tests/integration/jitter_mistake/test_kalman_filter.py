import unittest
import numpy as np

from jitter_smoothing.stream.implementations.kalman_filter import get_kalman


class TestKalmanFilterStream(unittest.TestCase):
    def test_smooth_normal_case_returns_ok(self):
        kalman_filter = get_kalman(np.array([[1, 2, 3], [4, 5, 6]]), q=0.05, r=1)
        motion_data = np.array(
            [
                [[2, 3, 4], [5, 6, 7]],
                [[3, 4, 5], [6, 7, 8]],
                [[4, 5, 6], [7, 8, 9]],
            ]
        )

        expected_result = np.array(
            [
                [[1.6970, 2.6970, 3.6970], [4.6970, 5.6970, 6.6970]],
                [[2.8512, 3.8512, 4.8512], [5.8512, 6.8512, 7.8512]],
                [[4.0302, 5.0302, 6.0302], [7.0302, 8.0302, 9.0302]],
            ]
        )
        for i, frame in enumerate(motion_data):
            kalman_filter.smooth_frame(frame)
            result = kalman_filter.get_last_smoothed_frame()
            np.testing.assert_array_almost_equal(result, expected_result[i], decimal=3)

    def test_smooth_two_frame2D_return_ok(self):
        motion_data = np.array([[[2, 3], [5, 6]], [[3, 4], [6, 7]]])

        expected_result = np.array(
            [[[1.6970, 2.6970], [4.6970, 5.6970]], [[2.8512, 3.8512], [5.8512, 6.8512]]]
        )
        kalman_filter = get_kalman(np.array([[1, 2], [4, 5]]), q=0.05, r=1)
        for i, frame in enumerate(motion_data):
            kalman_filter.smooth_frame(frame)
            result = kalman_filter.get_last_smoothed_frame()
            np.testing.assert_array_almost_equal(result, expected_result[i], decimal=3)
