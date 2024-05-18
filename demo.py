import numpy as np
from jitter_smoothing.batch.kalman_smoother import smooth
from jitter_smoothing.stream.implementations.single_exponentional_smoothing import (
    SingleExponentialSmoothingStream,
)
from rapid_change_detection.batch.detect_points_bones_length import detect_points
from rapid_change_detection.stream.implementations.detect_points_bones_length import (
    DetectPointsByBoneLength,
)
from models.media_pipe import MediaPipe
from rapid_change_correction.batch.interpolation import repair_by_interpolation
from rapid_change_correction.stream.implementations.single_exponential_smoothing import (
    SingleExponentionalSmoothingRepair,
)
from utils.loaders import load_mediapipe
from repair_stream import RepairStream

# BATCH processing


def batch():
    data = load_mediapipe(f"./computed_data/real_data/mediapipe_17/1.npy")
    removed_jitter_data = smooth(data)
    detected_points = detect_points(removed_jitter_data, MediaPipe.EDGES, treshold=0.08)
    rapid_change_corection = repair_by_interpolation(
        removed_jitter_data,
        detected_points,
        max_interpolate=10,
        previous_when_missing=True,
        extrapolate=False,
    )
    np.save(
        f"batch_result.npy",
        rapid_change_corection,
    )


batch()


# Stream processing


def stream():
    data = load_mediapipe(f"./computed_data/real_data/mediapipe_17/1.npy")
    jitter_component = SingleExponentialSmoothingStream(data[0], alpha=0.6)
    rapid_change_detection = DetectPointsByBoneLength(
        MediaPipe.EDGES, initial_frame=data[0], threshold=0.08
    )
    rapid_change_corection = SingleExponentionalSmoothingRepair(
        rapid_change_detection,
        data.shape[1],
        max_frames_to_compute=5,
        previous_when_missing=True,
    )
    repair_stream = RepairStream(jitter_component, rapid_change_corection)
    result = [data[0]]
    for frame in data[1:]:
        result.append(repair_stream.repair_frame(frame))
    np.save(f"stream_result.npy", result)


# stream()
