from utils.compute_change import compute_change
from utils.create_video import create_video
from models.hdm05 import HDM05
from jitter_smoothing.batch.kalman_smoother import smooth
from utils.loaders import load_HDM05_ground_truth, load_HDM05_errored_data

# Internaly we used coeficients to compute som you have to transfer it into cm then  with equation (x / 0.45) * 2.54


def compute():
    path_ground_truth = (
        "./computed_data/jitter/ground_truth"  # specify path to ground truth directory
    )
    path_errored_data = (
        "./computed_data/jitter/gaus"  # specify path to directory with errored data
    )
    error = 0.15  # specify error coeficient. Coeficients are [0.025, 0.05, 0.15, 0.25, 0.4, 0.5] which corespondent to centimeters [0.15,0.3,0.85,1.5,2.25,2.85]
    file_num = 1  # specify sequence
    ground_truth_frames = load_HDM05_ground_truth(
        path_ground_truth + f"/result{file_num}.json"
    )  # specify path to groud truth of sequence

    errored_data, _ = load_HDM05_errored_data(
        path_errored_data + f"/{error}-{file_num}.json"
    )  # specify path to errored sequence

    x = compute_change(ground_truth_frames, errored_data)
    result = smooth(errored_data)  # here specify algorithm to use
    y = compute_change(ground_truth_frames, result)

    print(
        x, y
    )  # results are in coeficients no in cm you can transfer it with equation (x / 0.45) * 2.54
    create_video(
        ground_truth_frames,
        errored_data,
        result,
        ground_truth_frames.shape[0],
        HDM05.EDGES,
        f"output_file.mp4",  # specify output file
    )


compute()
