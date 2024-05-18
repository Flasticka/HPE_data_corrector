class DummyDetectionComponenet:
    def __init__(self, num_of_frames, wrong_points) -> None:
        self.counter = 1
        self.frames = [[] for _ in range(num_of_frames + 1)]
        for i, joint in enumerate(wrong_points):
            for point in joint:
                self.frames[point].append(i)

    def check_frame(self, frame):
        to_return = self.frames[self.counter]
        self.counter += 1
        return to_return