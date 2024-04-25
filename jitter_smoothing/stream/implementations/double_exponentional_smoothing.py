import numpy as np

from exceptions.not_valid_dimensionality_exception import NotValidDimensionalityException
from ..interfaces.jitter_smoothing_interface import JitterSmoothingInterface


class DoubleExponentialSmoothingStream(JitterSmoothingInterface):

    def __init__(self, initial_frame, alpha=0.4, beta=0.1):
        self.current_state = np.array(initial_frame)
        self.level_component = np.array(initial_frame)
        self.trend_component = np.zeros(initial_frame.shape)
        self.alpha = alpha
        self.beta = beta

    def smooth_frame(self, frame):
        if len(frame.shape) != 2:
            raise NotValidDimensionalityException("frame", 2)
        new_level_component = self.alpha * frame + (1 - self.alpha) * (
            self.level_component - self.trend_component
        )
        new_trend_component = (
            self.beta * (new_level_component - self.level_component)
            + (1 - self.beta) * self.trend_component
        )
        self.current_state = new_level_component + new_trend_component
        self.level_component = new_level_component
        self.trend_component = new_trend_component

    def get_last_smoothed_frame(self):
        return self.current_state
