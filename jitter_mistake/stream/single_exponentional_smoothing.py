import numpy as np


class SingleExponentialSmoothingStream:

    def __init__(self, initial_frame, alpha=0.4):
        self.current_state = initial_frame
        self.alpha = alpha

    def smooth_frame(self, frame):
        new_state = []
        for i in range(self.current_state.shape[0]):
            curr_cords = []
            for j in range(self.current_state.shape[1]):
                curr_cords.append(self.alpha * frame[i][j] + (1 - self.alpha) * self.current_state[i][j])
            new_state.append(curr_cords)
        self.current_state = np.array(new_state)

    def get_last_smoothed_frame(self):
        return self.current_state
