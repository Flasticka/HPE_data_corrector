import numpy as np
from utils.compute_change import compute_change
from utils.create_video import create_video
from models.hdm05 import HDM05
from rapid_change_correction.stream.implementations.kalman import (
    KalmanFilterRepair,
)
from utils.loaders import load_HDM05_ground_truth, load_HDM05_errored_data

# Internaly we used coeficients to compute som you have to transfer it into cm then with equation (x / 0.45) * 2.54


def compute():
    path_ground_truth = (
        "./computed_data/jitter/ground_truth"  # specify path to ground truth directory
    )
    path_errored_data = "./computed_data/rapid_change_detection/results/"  # specify path to directory with errored data
    error = 4  # specify error coeficient. Coeficients are errors = [0.3, 0.5, 0.8, 1, 2, 4, 5] which corespondent to centimeters [1.5, 2.85, 4.5, 5.5, 11, 22.5, 28]
    prob_first = 0.01  # specify prob of first error
    prob_next = 0.5  # specify prob of next frame is errored after error
    file_num = 0  # specify sequence

    ground_truth_frames = load_HDM05_ground_truth(
        path_ground_truth + f"/result{file_num}.json"
    )  # specify path to groud truth of sequence

    errored_data, wrong = load_HDM05_errored_data(
        path_errored_data + f"result-{error}-{prob_first}-{prob_next}-{file_num}.json"
    )

    errored = compute_change(ground_truth_frames, errored_data)

    repaired_data = simulate_stream(
        errored_data, wrong
    )  # or use function in case of interpolation
    result = compute_change(ground_truth_frames, repaired_data)
    print(errored, result)

    create_video(
        ground_truth_frames,
        errored_data,
        repaired_data,
        ground_truth_frames.shape[0],
        HDM05.EDGES,
        f"output_file.mp4",  # specify output file
    )


class DummyDetectionComponenet:
    def __init__(self, num_of_frames, wrong_points) -> None:
        self.counter = 0
        self.frames = [[] for _ in range(num_of_frames)]
        for point in wrong_points:

            self.frames[point[1] - 1].append(point[0])

    def check_frame(self, frame):
        to_return = self.frames[self.counter]
        self.counter += 1
        return to_return


def simulate_stream(frames, wrong_points):
    result = []
    detection_component = DummyDetectionComponenet(frames.shape[0], wrong_points)
    repair_component = KalmanFilterRepair(
        detection_component,
        frames.shape[1],
        previous_when_missing=True,
        max_frames_to_compute=10,
    )  # specify algorithm
    for i, frame in enumerate(frames):
        result.append(repair_component.repair_frame(frame))

    return np.array(result)


compute()
