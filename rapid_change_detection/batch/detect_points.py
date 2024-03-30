import math


def detect_points(data, edges, num_of_frames_to_compute_from, exclude_edges):
    num_of_frames = data.shape[1]
    num_of_joints = data.shape[0]
    edges_to_use = edges - exclude_edges
    avgs = {}
    tresholds = {}
    result = [set() for _ in range(num_of_joints)]
    for i in range(num_of_frames):
        current = {}
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
                avgs[edge] = avgs.get(edge, 0) + res
            elif i == num_of_frames_to_compute_from:
                avgs[edge] /= num_of_frames_to_compute_from
                tresholds[edge] = avgs[edge]
            if i >= num_of_frames_to_compute_from and abs(avgs[edge] - res) > 0.25:
                result[first].add(i)
                result[second].add(i)
            current[edge] = res

    return list(map(lambda points: sorted(list(points)), result))
