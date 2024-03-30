import numpy as np
import scipy.interpolate as si


def get_missing_intervals(points_wrong_frames):
    if len(points_wrong_frames) == 0:
        return []
    points_wrong_frames = sorted(points_wrong_frames)
    result = []
    first = points_wrong_frames[0]
    second = points_wrong_frames[0]
    for i in range(1, len(points_wrong_frames)):
        if second == points_wrong_frames[i] - 1:
            second = points_wrong_frames[i]
        else:
            result.append((first, second))
            first = points_wrong_frames[i]
            second = points_wrong_frames[i]
    result.append((first, second))
    return result


def calculate_interpolation(data, start, end, a, b):
    result = []
    x = data[start:a].tolist() + data[b + 1 : end + 1].tolist()

    y = [i for i in range(start, a)] + [i for i in range(b + 1, end + 1)]

    if len(x) < 2:
        for j in range(a, b + 1):
            result.append(None)
        return result
    if len(x) == 2 or b - a < 5:
        i3 = si.interp1d(y, x, kind="linear")
    elif len(x) == 3 or b - a < 15:
        i3 = si.interp1d(y, x, kind="quadratic")
    else:
        i3 = si.interp1d(y, x, kind="cubic")
    for j in range(a, b + 1):
        result.append(i3(j).item())
    return result


def interpolate_frames(data, intervals, max_interpolate, num_to_look):
    result = []
    i = 0

    for j, interval in enumerate(intervals):

        a, b = interval
        start = max(i, a - num_to_look)
        upper_boundary = len(data) - 1
        if j + 1 < len(intervals):
            upper_boundary = intervals[j + 1][0]
        end = min(upper_boundary, b + num_to_look)
        while i < a:
            result.append(data[i])
            i += 1
        if b < len(data) - 1 and b - a < max_interpolate:
            result += calculate_interpolation(data, start, end, a, b)
            i = b + 1
        else:
            while i < b:
                result.append(None)
                i += 1

    while i < len(data):
        result.append(data[i])
        i += 1
    return result


def repair_by_interpolation(
    data, detected_points, max_interpolate=20, num_to_look_back=20
):
    result = []
    for i, joint in enumerate(data):
        transposed = joint.T
        intervals = get_missing_intervals(detected_points[i])
        joint_result = []
        for cord_frames in transposed:
            joint_result.append(
                interpolate_frames(
                    cord_frames, intervals, max_interpolate, num_to_look_back
                )
            )
        result.append(joint_result)
    return np.transpose(np.array(result), (0, 2, 1))
