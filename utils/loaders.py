import json
import numpy as np


def load_HDM05_ground_truth(path):
    with open(path, "r") as f:
        data = json.load(f)
        ground_truth_frames = np.transpose(np.array(data["data"]), (1, 0, 2))
    return ground_truth_frames


def load_HDM05_errored_data(path):
    with open(path, "r") as f:
        data = json.load(f)
        ground_truth_frames = np.transpose(np.array(data["data"]), (1, 0, 2))
        try:
            wrong_joints = set(map(lambda x: (x[0], x[1]), data["wrong"]))
        except KeyError:
            wrong_joints = set()
    return ground_truth_frames, wrong_joints


def load_mediapipe(path):
    return np.load(path)


def load_MHFormer_3D(path):
    return np.load(path)["reconstruction"]


def load_MHFormer_2D(path):
    return np.load(path)["reconstruction"][0]
