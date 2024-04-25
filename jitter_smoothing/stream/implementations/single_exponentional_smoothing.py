import numpy as np

from exceptions.not_valid_dimensionality_exception import NotValidDimensionalityException
from ..interfaces.jitter_smoothing_interface import JitterSmoothingInterface


class SingleExponentialSmoothingStream(JitterSmoothingInterface):

    def __init__(self, initial_frame, alpha=0.4):
        self.current_state = np.array(initial_frame)
        self.alpha = alpha

    def smooth_frame(self, frame):
        if len(frame.shape) != 2:
            raise NotValidDimensionalityException("frame", 2)
        self.current_state = self.alpha * frame + (1 - self.alpha) * self.current_state

    def get_last_smoothed_frame(self):
        return self.current_state
