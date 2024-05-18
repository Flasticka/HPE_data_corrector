import numpy as np
from models.hdm05 import HDM05
from rapid_change_detection.batch.detect_points_kalman import detect_points
from utils.compute_confusion_matrix import compute_confusion_matrix
from utils.loaders import load_HDM05_errored_data

# Internaly we used coeficients to compute som you have to transfer it into cm then


def compute():
    path_errored_data = "./computed_data/rapid_change_detection/results/"  # specify path to directory with errored data
    error = 4  # specify error coeficient. Coeficients are errors = [0.3, 0.5, 0.8, 1, 2, 4, 5] which corespondent to centimeters [1.5, 2.85, 4.5, 5.5, 11, 22.5, 28]
    prob_first = 0.01  # specify prob of first error
    prob_next = 0.5  # specify prob of next frame is errored after error
    file_num = 0  # specify sequence
    frames, wrong = load_HDM05_errored_data(
        path_errored_data + f"result-{error}-{prob_first}-{prob_next}-{file_num}.json"
    )
    bad_points = detect_points(np.array(frames))
    res = compute_confusion_matrix(
        HDM05.NUM_OF_JOINTS, frames.shape[0], wrong, bad_points
    )
    print(res)


compute()
