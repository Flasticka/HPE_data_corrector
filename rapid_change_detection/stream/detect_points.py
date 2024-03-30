import math


class DetectPointsByBoneLength:
    def __init__(self, edges, num_of_frames_to_compute_from, exclude_edges) -> None:
        self.edges_to_compute = edges - exclude_edges
        self.num_of_frames_to_compute_from = num_of_frames_to_compute_from
        self.number_of_computed = 0
        self.avgs = {}

    def detect_points_for_frame(self, frame):
        result = {}
        for edge in self.edges_to_compute:
            first, second = edge[0], edge[1]
            x_1 = frame[first][0]
            y_1 = frame[first][1]
            z_1 = frame[first][2]

            x_2 = frame[second][0]
            y_2 = frame[second][1]
            z_2 = frame[second][2]

            x = (x_1 - x_2) ** 2
            y = (y_1 - y_2) ** 2
            z = (z_1 - z_2) ** 2

            res = math.sqrt(x + y + z)

            if self.number_of_computed < self.num_of_frames_to_compute_from:
                self.avgs[edge] = self.avgs.get(edge, 0) + res
            elif self.number_of_computed == self.num_of_frames_to_compute_from:
                self.avgs[edge] /= self.num_of_frames_to_compute_from
                self.tresholds[edge] = self.avgs[edge]
            if (
                self.number_of_computed >= self.num_of_frames_to_compute_from
                and abs(self.avgs[edge] - res) > self.tresholds[edge]
            ):
                result.add(first)
                result.add(second)
        return result
