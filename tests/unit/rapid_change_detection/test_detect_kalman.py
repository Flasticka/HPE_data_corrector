import unittest
import numpy as np
from rapid_change_detection.batch.detect_points_kalman import (
    _initiate_kalman,
    _initiate_kalman2D,
    detect_points,
)


class TestDetectPoints(unittest.TestCase):
    def test_detect_points(self):
        data = np.array(
            [[[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]], [[4.0, 5.0, 6.0], [7.5, 8.5, 9.5]]]
        )
        expected_result = [[2], []]
        result = detect_points(data)
        self.assertEqual(result, expected_result)
