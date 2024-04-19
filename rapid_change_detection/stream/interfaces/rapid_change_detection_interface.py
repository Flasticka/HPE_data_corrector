from abc import ABC, abstractmethod


class RapidChangeDetectionInterface(ABC):
    @abstractmethod
    def check_frame(self, frame):
        """
        Method for finding faulty joints in frame
        :param frame: frame in which faulty joints are searched
        :return: set of faulty joints in frame
        """
        pass
