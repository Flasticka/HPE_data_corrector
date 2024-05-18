def transfer_to_tuples(result_data):
    result = set()
    for i in range(len(result_data)):
        for frame in result_data[i]:
            result.add((i, frame))
    return result


def compute_confusion_matrix(num_joints, num_of_frames, ground_truth, result_data):
    result_data = transfer_to_tuples(result_data)

    res = {"FN": 0, "FP": 0, "TN": 0, "TP": 0}
    for i in range(num_joints):
        for j in range(num_of_frames):
            j_f = (i, j)
            if j_f in ground_truth and j_f in result_data:
                res["TN"] += 1
                continue
            if j_f not in ground_truth and j_f in result_data:
                res["FN"] += 1
                continue
            if j_f in ground_truth and j_f not in result_data:
                res["FP"] += 1
                continue
            if j_f not in ground_truth and j_f not in result_data:
                res["TP"] += 1
                continue
    res["sum"] = res["FN"] + res["TN"] + res["TP"] + res["FP"]
    res["accuracy"] = round((res["TN"] + res["TP"]) / res["sum"], 3)
    res["error rate"] = round(1 - res["accuracy"], 3)
    res["true positive rate"] = round(res["TP"] / (res["TP"] + res["FN"]), 3)
    res["true negative rate"] = round(res["TN"] / (res["TN"] + res["FP"]), 3)
    return res
