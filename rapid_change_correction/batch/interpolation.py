import numpy as np
import scipy.interpolate as si


def _get_missing_intervals(points_wrong_frames):
    if not points_wrong_frames:
        return []

    points_wrong_frames = sorted(points_wrong_frames)
    result = []
    first = second = points_wrong_frames[0]

    for frame in points_wrong_frames[1:]:
        if second == frame - 1:
            second = frame
        else:
            result.append((first, second))
            first = second = frame

    result.append((first, second))
    return result


def _calculate_interpolation(data, start, end, a, b, previous_when_missing):
    result = []
    x = np.concatenate((data[start:a], data[b + 1 : end + 1]))

    y = np.concatenate((np.arange(start, a), np.arange(b + 1, end + 1)))

    if len(x) < 2:
        for j in range(a, b + 1):
            result.append(data[j] if previous_when_missing else None)
        return result

    if len(x) == 2 or b - a < 5:
        i3 = si.interp1d(y, x, kind="linear")
    elif len(x) == 3 or b - a < 10:
        i3 = si.interp1d(y, x, kind="quadratic")
    else:
        i3 = si.interp1d(y, x, kind="cubic")

    for j in range(a, b + 1):
        result.append(i3(j).item())
    return result


def _calculate_extrapolation(
    data, start, end, a, b, previous_when_missing, max_extrapolate
):
    result = []
    x = data[start:a]
    y = np.arange(start, a)
    if len(x) < 2:
        for j in range(a, a + max_extrapolate):
            result.append(data[j] if previous_when_missing else None)
    else:
        extrapolate_func = si.interp1d(y, x, fill_value="extrapolate")
        for j in range(a, a + max_extrapolate):
            result.append(extrapolate_func(j).item())
    for j in range(a + max_extrapolate, (b - max_extrapolate) + 1):
        result.append(data[j] if previous_when_missing else None)
    x = data[b + 1 : end + 1]
    y = np.arange(b + 1, end + 1)

    if len(x) < 2:
        for j in range((b - max_extrapolate) + 1, b + 1):
            result.append(data[j] if previous_when_missing else None)
    else:
        extrapolate_func = si.interp1d(y, x, fill_value="extrapolate")
        for j in range((b - max_extrapolate) + 1, b + 1):
            result.append(extrapolate_func(j).item())
    return result


def _interpolate_frames(
    data, intervals, max_interpolate, num_to_look, extrapolate, previous_when_missing
):
    result = []
    i = 0

    for j, (a, b) in enumerate(intervals):
        a -= 1
        b -= 1

        last_interval_end = -1
        next_interval_start = len(data)
        if j != 0:
            last_interval_end = intervals[j - 1][1] - 1
        if j + 1 < len(intervals):
            next_interval_start = intervals[j + 1][0] - 1
        if a - 2 > last_interval_end + 1:
            a -= 2
        if b + 2 < next_interval_start - 2:
            b += 2
        start = max(last_interval_end + 1, a - num_to_look)

        end = min(
            next_interval_start - 1,
            b + num_to_look,
        )

        while i < a:
            result.append(data[i])
            i += 1
        if b - a > max_interpolate and extrapolate:
            result.extend(
                _calculate_extrapolation(
                    data, start, end, a, b, previous_when_missing, max_interpolate // 2
                )
            )
            i = b + 1
        elif (
            b < len(data) - 1
            and end <= len(data) - 1
            and b - a <= max_interpolate
            and a > 0
        ):
            result.extend(
                _calculate_interpolation(data, start, end, a, b, previous_when_missing)
            )
            i = b + 1
        else:
            while i < b + 1:
                result.append(data[i] if previous_when_missing else None)
                i += 1
    result.extend(data[i:])
    return result


def repair_by_interpolation(
    data,
    detected_points,
    max_interpolate=5,
    num_to_look_back=5,
    extrapolate=True,
    previous_when_missing=False,
):
    data = np.transpose(data, (1, 0, 2))
    result = []
    for joint in data:
        transposed = np.transpose(joint)
        intervals = _get_missing_intervals(detected_points[len(result)])
        joint_result = []
        for cord_frames in transposed:
            res = _interpolate_frames(
                cord_frames,
                intervals,
                max_interpolate,
                num_to_look_back,
                extrapolate,
                previous_when_missing,
            )
            joint_result.append(res)
        result.append(joint_result)

    return np.transpose(np.array(result), (2, 0, 1))
