
def compute_change(ground_truth_data, result_data):
    results = {}
    total = 0
    total_avg = 0
    for i, joint in enumerate(ground_truth_data):
        sum_joint_mistake = 0
        frame_count = 0
        missing = 0
        for j, frame in enumerate(joint):
            if len(result_data[i][j]) > 0:
                frame_count += 1
                sum_joint_mistake += abs(result_data[i][j][0] - frame[0])
                sum_joint_mistake += abs(result_data[i][j][1] - frame[1])
                sum_joint_mistake += abs(result_data[i][j][2] - frame[2])
            else:
                missing += 1
        results[i + 1] = {"sum_error": sum_joint_mistake,
                          "avg_for_joint": sum_joint_mistake / frame_count,
                          "avg_for_joint_cord": (sum_joint_mistake / frame_count) / 3,
                          "missing": missing}
        total_avg += sum_joint_mistake / frame_count
        total += sum_joint_mistake

    results["total"] = {"total_average": total_avg, "total": total}

    return results
