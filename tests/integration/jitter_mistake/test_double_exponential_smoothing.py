import unittest
import numpy as np

from jitter_smoothing.stream.implementations.double_exponentional_smoothing import (
    DoubleExponentialSmoothingStream,
)


class TestDoubleExponentialSmoothingStream(unittest.TestCase):
    def test_smooth_normal_case_returns_ok(self):
        double_exponential = DoubleExponentialSmoothingStream(
            np.array([[1, 2, 3], [4, 5, 6]]), alpha=0.4, beta=0.1
        )
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
        for i, frame in enumerate(motion_data):
            double_exponential.smooth_frame(frame)
            result = double_exponential.get_last_smoothed_frame()
            np.testing.assert_array_almost_equal(result, expected_result[i], decimal=3)
