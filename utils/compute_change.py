from math import sqrt
import numpy as np


def compute_change(ground_truth_data, result_data):
    ground_truth_data = np.transpose(ground_truth_data, (1, 0, 2))
    result_data = np.transpose(result_data, (1, 0, 2))
    results = {}
    total = 0
    total_frame_joint_count = 0
    missing = 0
    for i, joint in enumerate(ground_truth_data):
        sum_joint_mistake = 0
        for j, frame in enumerate(joint):
            if result_data[i][j][0] is not None:
                total_frame_joint_count += 1
                lenghts = 0
                lenghts += (result_data[i][j][0] - frame[0]) ** 2
                lenghts += (result_data[i][j][1] - frame[1]) ** 2
                lenghts += (result_data[i][j][2] - frame[2]) ** 2
                sum_joint_mistake += sqrt(lenghts)
            else:
                missing += 1

        total += sum_joint_mistake

    results["total"] = {
        "total_average_per_frame": total / total_frame_joint_count,
        "total": total,
        "missing": missing,
    }

    return results
