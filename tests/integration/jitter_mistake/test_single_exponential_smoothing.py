import unittest
import numpy as np

from exceptions.not_valid_dimensionality_exception import NotValidDimensionalityException
from jitter_smoothing.stream.implementations.single_exponentional_smoothing import SingleExponentialSmoothingStream


class TestSingleExponentialSmoothingStream(unittest.TestCase):
    def test_smooth_normal_case_returns_ok(self):
        single_exponential = SingleExponentialSmoothingStream(np.array([[1, 2, 3], [4, 5, 6]]), alpha=0.4)
        motion_data = np.array(
            [
                [[2, 3, 4], [5, 6, 7]],
                [[3, 4, 5], [6, 7, 8]],
                [[4, 5, 6], [7, 8, 9]],
            ]
        )

        expected_result = np.array(
            [
                [[1.4, 2.4, 3.4], [4.4, 5.4, 6.4]],
                [[2.04, 3.04, 4.04], [5.04, 6.04, 7.04]],
                [[2.824, 3.824, 4.824], [5.824, 6.824, 7.824]],
            ]
        )
        for i, frame in enumerate(motion_data):
            single_exponential.smooth_frame(frame)
            result = single_exponential.get_last_smoothed_frame()
            np.testing.assert_array_almost_equal(result, expected_result[i], decimal=3)

    def test_smooth_no_frame_raise_exception(self):
        frame = np.array([])
        single_exponential = SingleExponentialSmoothingStream(np.array([[1, 2, 3]]), alpha=0.4)

        try:
            single_exponential.smooth_frame(frame)
        except NotValidDimensionalityException as err:
            self.assertEqual(
                "Invalid dimensionality of variable 'frame', expected dimensionality 2.",
                str(err),
            )
