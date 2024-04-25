from abc import ABC, abstractmethod


class JitterSmoothingInterface(ABC):
    @abstractmethod
    def smooth_frame(self, frame):
        """
        Method for smoothing frame.
        :param frame: frame to be smooth
        :return: None
        """
        pass

    @abstractmethod
    def get_last_smoothed_frame(self):
        """
        Method for returning last smoothed frame. Typically called after smooth_frame method.
        :return: last smoothed frame
        """
        pass
