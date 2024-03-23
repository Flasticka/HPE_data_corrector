
def compute_avg_err_length(data, num_frames, num_of_joints):
    data_formatted = set(map(lambda x: (x[0], x[1]), data))
    summ = 0
    count = 0
    for i in range(num_of_joints):
        curr = 0
        for j in range(num_frames):
            if (i, j) in data_formatted:
                curr += 1
                continue
            if curr > 0:
                summ += curr
                curr = 0
                count += 1
    return round(summ / count, 3)