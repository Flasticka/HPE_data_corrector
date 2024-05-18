class RepairStream:
    def __init__(self, jitter_remove_algorithm, rapid_change_algorithm) -> None:
        self.jitter_remove_algorithm = jitter_remove_algorithm
        self.rapid_change_algorithm = rapid_change_algorithm
        self.counter = 1

    def repair_frame(self, frame):
        self.jitter_remove_algorithm.smooth_frame(frame)
        removed_jitter_frame = self.jitter_remove_algorithm.get_last_smoothed_frame()

        repaired_rapid_change = self.rapid_change_algorithm.repair_frame(
            removed_jitter_frame
        )
        self.counter += 1
        return repaired_rapid_change
