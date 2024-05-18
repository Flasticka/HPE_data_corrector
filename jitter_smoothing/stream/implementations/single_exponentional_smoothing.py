import numpy as np
from ..interfaces.jitter_smoothing_interface import JitterSmoothingInterface


class SingleExponentialSmoothingStream(JitterSmoothingInterface):

    def __init__(self, initial_frame, alpha=0.4):
        self.current_state = np.array(initial_frame)
        self.alpha = alpha

    def smooth_frame(self, frame):
        if np.isnan(self.current_state).any():
            self.current_state = frame
        else:
            self.current_state = (
                self.alpha * frame + (1 - self.alpha) * self.current_state
            )

    def get_last_smoothed_frame(self):
        return self.current_state
