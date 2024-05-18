import unittest
import numpy as np
from rapid_change_detection.batch.detect_points_bones_length import detect_points


class TestDetectPoints(unittest.TestCase):
    def test_detect_points(self):
        data = np.transpose(
            np.array(
                [
                    [[1, 2, 3], [1, 2, 3]],
                    [[2, 3, 4], [5, 6, 7]],
                    [[3, 4, 5], [6, 7, 8]],
                    [[4, 5, 6], [7, 8, 9]],
                ]
            ),
            (1, 0, 2),
        )
        edges = {(0, 1), (2, 3)}
        num_of_frames_to_compute_from = 1
        exclude_edges = set()
        threshold = 0.1
        expected_result = [[2], [2], [], []]
        result = detect_points(
            data, edges, num_of_frames_to_compute_from, exclude_edges, threshold
        )
        self.assertEqual(result, expected_result)
