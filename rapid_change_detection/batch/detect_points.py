import math
import numpy as np


def detect_points(
    data, edges, num_of_frames_to_compute_from, exclude_edges, coef_max_change=0.2
):
    num_of_frames = data.shape[1]
    num_of_joints = data.shape[0]
    edges_to_use = edges - exclude_edges
    first_n = {edge: [] for edge in edges_to_use}
    avgs = {}
    tresholds = {}
    result = [set() for _ in range(num_of_joints)]
    for i in range(num_of_frames):
        for edge in edges_to_use:

            first, second = edge[0], edge[1]
            x_1 = data[first][i][0]
            y_1 = data[first][i][1]
            z_1 = data[first][i][2]

            x_2 = data[second][i][0]
            y_2 = data[second][i][1]
            z_2 = data[second][i][2]

            x = (x_1 - x_2) ** 2
            y = (y_1 - y_2) ** 2
            z = (z_1 - z_2) ** 2

            res = math.sqrt(x + y + z)

            if i < num_of_frames_to_compute_from:
                first_n[edge].append(res)
            elif i == num_of_frames_to_compute_from:
                avgs[edge] = np.median(first_n[edge])
                tresholds[edge] = avgs[edge] * coef_max_change
            if (
                i >= num_of_frames_to_compute_from
                and abs(res - avgs[edge]) > tresholds[edge]
            ):
                result[first].add(i)
                result[second].add(i)
    return list(map(lambda points: sorted(list(points)), result))
