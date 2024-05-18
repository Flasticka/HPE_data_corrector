import unittest
import numpy as np
from rapid_change_correction.batch.interpolation import (
    repair_by_interpolation,
)
from . import DummyDetectionComponenet


class TestCorrectPoints(unittest.TestCase):
    def test_correct_one_point(self):
        data = np.array(
            [
                [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]],
                [[1.5, 2.5, 3.5], [7.0, 8.0, 9.0]],
                [[2, 3, 4], [7.0, 8.0, 9.0]],
                [[4.0, 5.0, 6.0], [7.5, 8.5, 9.5]],
                [[3, 4, 5], [7.0, 8.0, 9.0]],
            ]
        )
        result = repair_by_interpolation(data, [[4], []])
        expected_result = [
            [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]],
            [[1.5, 2.5, 3.5], [7.0, 8.0, 9.0]],
            [[2.0, 3.0, 4.0], [7.0, 8.0, 9.0]],
            [[2.5, 3.5, 4.5], [7.5, 8.5, 9.5]],
            [[3.0, 4.0, 5.0], [7.0, 8.0, 9.0]],
        ]

        np.testing.assert_array_almost_equal(result, expected_result, decimal=3)

    def test_correct_two_point(self):
        data = np.array(
            [
                [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]],
                [[1.5, 2.5, 3.5], [7.0, 8.0, 9.0]],
                [[2, 3, 4], [7.0, 8.0, 9.0]],
                [[4.0, 5.0, 6.0], [7.5, 8.5, 9.5]],
                [[4.5, 5.5, 6.6], [8, 9, 10]],
                [[3.5, 4.5, 5.5], [7.0, 8.0, 9.0]],
            ]
        )
        result = repair_by_interpolation(data, [[4, 5], []])
        expected_result = [
            [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]],
            [[1.5, 2.5, 3.5], [7.0, 8.0, 9.0]],
            [[2.0, 3.0, 4.0], [7.0, 8.0, 9.0]],
            [[2.5, 3.5, 4.5], [7.5, 8.5, 9.5]],
            [[3.0, 4.0, 5.0], [8.0, 9.0, 10.0]],
            [[3.5, 4.5, 5.5], [7.0, 8.0, 9.0]],
        ]

        np.testing.assert_array_almost_equal(result, expected_result, decimal=3)
