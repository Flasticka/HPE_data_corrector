from abc import ABC, abstractmethod


class RapidChangeRepairStreamInterface(ABC):
    def __init__(
        self,
        detection_component,
    ) -> None:
        self.detection_component = detection_component
        super().__init__()

    @abstractmethod
    def repair_frame(self, frame):
        """
        Method for repairing frame.
        :param frame: frame which is being repaired
        :return: repaired frame
        """
        pass
