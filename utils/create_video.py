import cv2
import numpy as np
import json


def create_video(
    ground_truth_data, errored_data, repaired_data, num_frames, edges, res_file
):
    ground_truth_data = np.transpose(ground_truth_data, (1, 0, 2))
    errored_data = np.transpose(errored_data, (1, 0, 2))
    repaired_data = np.transpose(repaired_data, (1, 0, 2))
    # Create a blank image with the same dimensions as the video
    width, height = 600, 700  # Set the dimensions according to your requirements
    blank_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(
        "H", "2", "6", "4"
    )  # You can use other codecs like 'MJPG' or 'H264'
    output_video = cv2.VideoWriter(res_file, fourcc, 30, (width, height))

    x_plus = 300
    y_plus = 300
    # Loop through frames
    for i in range(num_frames):
        # Create a blank frame
        frame = np.copy(blank_image)

        # Draw skeleton on the errored
        for edge in edges:
            try:
                first, second = edge[0], edge[1]
                x1, x2, y1, y2 = None, None, None, None
                if errored_data[first][i][0] is not None:
                    x1, y1 = (-errored_data[first][i][0] * 20) + 300, (
                        -errored_data[first][i][1] * 20
                    ) + 300
                if errored_data[second][i][0] is not None:
                    x2, y2 = (-errored_data[second][i][0] * 20) + 300, (
                        -errored_data[second][i][1] * 20
                    ) + 300

                if x1 and x2:
                    # Scale and draw the line
                    cv2.line(
                        frame,
                        (round(x1), round(y1)),
                        (round(x2), round(y2)),
                        (0, 0, 255),
                        3,
                    )
            except IndexError:
                break
        # Draw ground truth skeleton on the frame

        for edge in edges:
            try:
                first, second = edge[0], edge[1]
                x1, x2, y1, y2 = None, None, None, None
                if ground_truth_data[first][i][0] is not None:
                    x1, y1 = (-ground_truth_data[first][i][0] * 20) + x_plus, (
                        -ground_truth_data[first][i][1] * 20
                    ) + y_plus
                if ground_truth_data[second][i][0] is not None:
                    x2, y2 = (-ground_truth_data[second][i][0] * 20) + x_plus, (
                        -ground_truth_data[second][i][1] * 20
                    ) + y_plus

                if x1 and x2:
                    # Scale and draw the line
                    cv2.line(
                        frame,
                        (round(x1), round(y1)),
                        (round(x2), round(y2)),
                        (0, 255, 0),
                        3,
                    )
            except IndexError:
                break
        for edge in edges:
            try:
                first, second = edge[0], edge[1]
                x1, x2, y1, y2 = None, None, None, None
                if repaired_data[first][i][0] is not None:
                    x1, y1 = (-repaired_data[first][i][0] * 20) + x_plus, (
                        -repaired_data[first][i][1] * 20
                    ) + y_plus
                if repaired_data[second][i][0] is not None:
                    x2, y2 = (-repaired_data[second][i][0] * 20) + x_plus, (
                        -repaired_data[second][i][1] * 20
                    ) + y_plus

                if x1 and x2:
                    # Scale and draw the line
                    cv2.line(
                        frame,
                        (round(x1), round(y1)),
                        (round(x2), round(y2)),
                        (255, 0, 0),
                        3,
                    )
            except IndexError:
                break

        # Write the frame to the output video
        output_video.write(frame)

        # Display the frame
        cv2.imshow("Skeleton Video", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    # Release the video writer object
    output_video.release()
    cv2.destroyAllWindows()
