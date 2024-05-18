import unittest
import numpy as np
from rapid_change_detection.batch.detect_points_double import (
    _DoubleExponentialSmoothingStream,
    detect_points,
)


class TestDoubleExponentialSmoothingStream(unittest.TestCase):
    def test_initialization(self):
        initial_frame = np.array([1, 2, 3])
        des = _DoubleExponentialSmoothingStream(initial_frame)
        self.assertTrue(np.array_equal(des.current_state, initial_frame))
        self.assertTrue(np.array_equal(des.level_component, initial_frame))
        self.assertTrue(
            np.array_equal(des.trend_component, np.zeros_like(initial_frame))
        )
        self.assertTrue(
            np.array_equal(des.post_predict_current_state, np.zeros_like(initial_frame))
        )
        self.assertTrue(
            np.array_equal(
                des.post_predict_level_component, np.zeros_like(initial_frame)
            )
        )
        self.assertTrue(
            np.array_equal(
                des.post_predict_trend_component, np.zeros_like(initial_frame)
            )
        )
        self.assertEqual(des.alpha, 0.6)
        self.assertEqual(des.beta, 0.1)

    def test_predict_joint(self):
        initial_frame = np.array([1, 2, 3])
        des = _DoubleExponentialSmoothingStream(initial_frame)
        frame = np.array([4, 5, 6])
        expected_predicted = [2.98, 3.98, 4.98]
        des.predict_joint(frame)
        self.assertTrue(np.allclose(des.get_last_predicted_frame(), expected_predicted))

    def test_update(self):
        initial_frame = np.array([1, 2, 3])
        des = _DoubleExponentialSmoothingStream(initial_frame)
        frame = np.array([4, 5, 6])
        des.update(frame)
        self.assertTrue(np.array_equal(des.level_component, frame))

    def test_get_last_predicted_frame(self):
        initial_frame = np.array([1, 2, 3])
        des = _DoubleExponentialSmoothingStream(initial_frame)
        frame = np.array([4, 5, 6])
        des.predict_joint(frame)
        self.assertTrue(
            np.array_equal(
                des.get_last_predicted_frame(), np.array(des.post_predict_current_state)
            )
        )


class TestDetectPoints(unittest.TestCase):
    def test_detect_points(self):
        data = np.array(
            [[[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]], [[4.0, 5.0, 6.0], [7.5, 8.5, 9.5]]]
        )
        expected_result = [[2], []]
        result = detect_points(data)
        self.assertEqual(result, expected_result)
