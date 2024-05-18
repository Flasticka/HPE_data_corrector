import unittest
import numpy as np
from rapid_change_correction.stream.implementations.double_exponential_smoothing import (
    DoubleExponentialSmoothingRepair,
)
from . import DummyDetectionComponenet


class TestCorrectPoints(unittest.TestCase):
    def test_correct_one_point(self):
        data = np.array(
            [[[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]], [[4.0, 5.0, 6.0], [7.5, 8.5, 9.5]]]
        )
        detection_component = DummyDetectionComponenet(2, [[2], []])
        result = DoubleExponentialSmoothingRepair(detection_component, 2)
        res = []
        for frame in data:
            res.append(result.repair_frame(frame))
        expected_result = [
            [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]],
            [[1.66, 2.66, 3.66], [7.5, 8.5, 9.5]],
        ]

        np.testing.assert_array_almost_equal(res, expected_result, decimal=3)

    def test_correct_two_point(self):
        data = np.array(
            [
                [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]],
                [[4.0, 5.0, 6.0], [7.5, 8.5, 9.5]],
                [[4.5, 5.5, 6.6], [8, 9, 10]],
            ]
        )
        detection_component = DummyDetectionComponenet(3, [[2, 3], []])
        result = DoubleExponentialSmoothingRepair(detection_component, 2)
        res = []
        for frame in data:
            res.append(result.repair_frame(frame))
        expected_result = [
            [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]],
            [[1.66, 2.66, 3.66], [7.5, 8.5, 9.5]],
            [[1.77, 2.77, 3.792], [8.0, 9.0, 10.0]],
        ]

        np.testing.assert_array_almost_equal(res, expected_result, decimal=3)
