import cv2
import numpy as np
import json


def create_video(ground_truth_data, errored_data, repaired_data, num_frames, edges, res_file):
    # Create a blank image with the same dimensions as the video
    width, height = 800, 1000  # Set the dimensions according to your requirements
    blank_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(
        *"XVID"
    )  # You can use other codecs like 'MJPG' or 'H264'
    output_video = cv2.VideoWriter(res_file, fourcc, 30, (width, height))

    # Loop through frames
    for i in range(num_frames):
        # Create a blank frame
        frame = np.copy(blank_image)

        # Draw skeleton on the errored
        for edge in edges:
            try:
                first, second = edge[0], edge[1]
                x1, x2, y1, y2 = None, None, None, None
                if len(errored_data[first][i]) > 0:
                    x1, y1 = (-errored_data[first][i][0] * 20) + 300, (
                            -errored_data[first][i][1] * 20
                    ) + 300
                if len(errored_data[second][i]) > 0:
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
                        2,
                    )
            except IndexError:
                break
        # Draw ground truth skeleton on the frame

        for edge in edges:
            try:
                first, second = edge[0], edge[1]
                x1, x2, y1, y2 = None, None, None, None
                if len(ground_truth_data[first][i]) > 0:
                    x1, y1 = (-ground_truth_data[first][i][0] * 20) + 300, (
                            -ground_truth_data[first][i][1] * 20
                    ) + 300
                if len(ground_truth_data[second][i]) > 0:
                    x2, y2 = (-ground_truth_data[second][i][0] * 20) + 300, (
                            -ground_truth_data[second][i][1] * 20
                    ) + 300

                if x1 and x2:
                    # Scale and draw the line
                    cv2.line(
                        frame,
                        (round(x1), round(y1)),
                        (round(x2), round(y2)),
                        (0, 255, 0),
                        2,
                    )
            except IndexError:
                break
        for edge in edges:
            try:
                first, second = edge[0], edge[1]
                x1, x2, y1, y2 = None, None, None, None
                if len(repaired_data[first][i]) > 0:
                    x1, y1 = (-repaired_data[first][i][0] * 20) + 300, (
                            -repaired_data[first][i][1] * 20
                    ) + 300
                if len(repaired_data[second][i]) > 0:
                    x2, y2 = (-repaired_data[second][i][0] * 20) + 300, (
                            -repaired_data[second][i][1] * 20
                    ) + 300

                if x1 and x2:
                    # Scale and draw the line
                    cv2.line(
                        frame,
                        (round(x1), round(y1)),
                        (round(x2), round(y2)),
                        (255, 0, 0),
                        2,
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
