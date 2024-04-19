import math
import numpy as np


class DetectPointsByBoneLength:
    def __init__(
        self,
        edges,
        num_of_frames_to_compute_from=5,
        exclude_edges=set(),
        coef_max_change=0.2,
    ) -> None:
        self.edges_to_compute = edges - exclude_edges
        self.num_of_frames_to_compute_from = num_of_frames_to_compute_from
        self.number_of_computed = 0
        self.coef_max_change = coef_max_change
        self.avgs = {}
        self.tresholds = {}
        self.first_n = {edge: [] for edge in self.edges_to_compute}

    def check_frame(self, frame):
        result = set()
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
                self.first_n[edge].append(res)
            elif self.number_of_computed == self.num_of_frames_to_compute_from:
                self.avgs[edge] = np.median(self.first_n[edge])
                self.tresholds[edge] = self.avgs[edge] * self.coef_max_change
            if (
                self.number_of_computed >= self.num_of_frames_to_compute_from
                and abs(self.avgs[edge] - res) > self.tresholds[edge]
            ):
                result.add(first)
                result.add(second)
        self.number_of_computed += 1
        return result
