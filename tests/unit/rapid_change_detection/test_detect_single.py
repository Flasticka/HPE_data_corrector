import unittest
import numpy as np
from rapid_change_detection.batch.detect_points_single import (
    _SingleExponentialSmoothingStream,
    detect_points,
)


class TestSingleExponentialSmoothingStream(unittest.TestCase):
    def test_initialization(self):
        initial_frame = np.array([1, 2, 3])
        ses = _SingleExponentialSmoothingStream(initial_frame)
        self.assertTrue(np.array_equal(ses.current_state, initial_frame))
        self.assertTrue(np.array_equal(ses.predicted, initial_frame))
        self.assertEqual(ses.alpha, 0.2)

    def test_predict_frame(self):
        initial_frame = np.array([1, 2, 3])
        ses = _SingleExponentialSmoothingStream(initial_frame)
        frame = np.array([4, 5, 6])
        ses.predict_frame(frame)
        expected_predicted = np.array([1.6, 2.6, 3.6])
        self.assertTrue(np.allclose(ses.predicted, expected_predicted))

    def test_update(self):
        initial_frame = np.array([1, 2, 3])
        ses = _SingleExponentialSmoothingStream(initial_frame)
        frame = np.array([4, 5, 6])
        ses.update(frame)
        self.assertTrue(np.array_equal(ses.current_state, frame))

    def test_get_last_predict_frame(self):
        initial_frame = np.array([1, 2, 3])
        ses = _SingleExponentialSmoothingStream(initial_frame)
        frame = np.array([4, 5, 6])
        ses.predict_frame(frame)
        self.assertTrue(np.array_equal(ses.get_last_predict_frame(), ses.predicted))


class TestDetectPoints(unittest.TestCase):
    def test_detect_points(self):
        data = np.array(
            [[[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]], [[4.0, 5.0, 6.0], [7.5, 8.5, 9.5]]]
        )
        expected_result = [[2], []]
        result = detect_points(data)
        self.assertEqual(result, expected_result)
