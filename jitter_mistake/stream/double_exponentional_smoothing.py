import numpy as np


class DoubleExponentialSmoothingStream:

    def __init__(self, initial_frame, alpha=0.5, beta=0.2):
        self.current_state = initial_frame
        self.level_component = initial_frame
        self.trend_component = np.zeros(initial_frame.shape)
        self.alpha = alpha
        self.beta = beta

    def smooth_frame(self, frame):
        new_state = np.zeros(self.current_state.shape)
        new_level_component = np.zeros(self.current_state.shape)
        new_trend_component = np.zeros(self.current_state.shape)
        for i in range(self.current_state.shape[0]):
            for j in range(self.current_state.shape[1]):
                curr_level_component = self.alpha * frame[i][j] + (1 - self.alpha) * (
                    self.level_component[i][j] - self.trend_component[i][j]
                )
                curr_trend_component = (
                    self.beta * (curr_level_component - self.level_component[i][j])
                    + (1 - self.beta) * self.trend_component[i][j]
                )
                new_state[i][j] = curr_level_component + curr_trend_component
                new_level_component[i][j] = curr_level_component
                new_trend_component[i][j] = curr_trend_component
        self.current_state = new_state
        self.level_component = new_level_component
        self.trend_component = new_trend_component

    def get_last_smoothed_frame(self):
        return self.current_state
