import cv2
from enum import Enum
import time

from pattern_type import PatternType
from pattern_finder import PatternFinder
from camera_calibrator import CameraCalibrator
from geometry import line_intersection
from drawing_helpers import draw_horizontal_line, draw_vertical_line, draw_numbered_points


class Mode(Enum):
    Initial = 0,
    Calibration = 1,
    Calibrated = 2


def visualise_taking_sample(frame):
    cv2.circle(frame, (frame.shape[1]//10, frame.shape[0]//10), frame.shape[0]//25, (255,255,255), -1, cv2.LINE_AA)
    cv2.circle(frame, (frame.shape[1]//10*9, frame.shape[0]//10), frame.shape[0]//25, (255,255,255), -1, cv2.LINE_AA)
    cv2.circle(frame, (frame.shape[1]//10, frame.shape[0]//10*9), frame.shape[0]//25, (255,255,255), -1, cv2.LINE_AA)
    cv2.circle(frame, (frame.shape[1]//10*9, frame.shape[0]//10*9), frame.shape[0]//20, (255,255,255), -1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    pattern_dims = (5, 7)
    pattern_type = PatternType.Checkerboard
    pattern_finder = PatternFinder(pattern_type, pattern_dims)
    pattern_finder.start()
    calibrator = CameraCalibrator(pattern_type, pattern_dims)
    last_calibration_sample = None
    mode = Mode.Initial

    map_x, map_y, roi = None, None, None

    while True:
        success, frame = cap.read()
        if not success:
            continue  # Let's just try again

        if not pattern_finder.recognition_in_progress:
            #Get results and start a next recognition
            pattern_found = pattern_finder.pattern_found
            pattern_points = pattern_finder.pattern_points
            pattern_finder.start_pattern_recognition(frame)

            if pattern_found:

                if mode == Mode.Calibration:
                    if last_calibration_sample is None or time.time() - last_calibration_sample > 0.2:
                        last_calibration_sample = time.time()
                        calibrator.add_sample(pattern_points)
                        visualise_taking_sample(frame)

                    if calibrator.number_of_samples > 100:
                        cv2.rectangle(
                            frame,
                            (0, 0),
                            (frame.shape[1], frame.shape[0]),
                            (255, 255, 255),
                            -1
                        )
                        cv2.putText(frame, "CALIBRATING...", (frame.shape[1]//3, frame.shape[0]//2), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.imshow('frame', frame)
                        cv2.waitKey(20)

                        ret, map_x, map_y, roi = calibrator.calibrate(frame.shape)
                        print("Calibration error: {}".format(ret))
                        mode = Mode.Calibrated

                cv2.drawChessboardCorners(frame, pattern_dims, pattern_points, pattern_found)

                #Find four corners of the board and use then to nif horizon and visualise it
                corners = find_checkerboard_corners(pattern_points, pattern_dims)
                draw_numbered_points(frame, corners)
                intersection = line_intersection(corners[0:2], corners[2:4])
                #Draw lines to intersection (horizon)
                for corner in corners:
                    cv2.line(frame, corner, intersection, (255, 0, 0), 1, cv2.LINE_AA)
                horizon = intersection[1]
                draw_horizontal_line(frame, horizon, (200, 0, 0), 2, cv2.LINE_AA)
                draw_vertical_line(frame, intersection[0], (200, 0, 0), 1, cv2.LINE_AA)

        draw_vertical_line(frame, frame.shape[1]//2, (100, 0, 0), 1, cv2.LINE_AA)


        # Display the resulting frame
        cv2.imshow('frame', frame)

        if mode == Mode.Calibrated:
            dst = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
            cv2.imshow('undistorted', dst)
            #x, y, w, h = roi
            #dst = dst[y:y+h, x:x+w]
            #cv2.imshow('undistorted_cropped', dst)

        key_no = cv2.waitKey(30) & 0xFF
        if key_no == ord('q'):
            print("Quitting...")
            break
        if key_no == ord('c'):
            print("Calibrate started")
            calibrator.clear()
            mode = Mode.Calibration
    #End (while True)
    cap.release()
    cv2.destroyAllWindows()


def find_checkerboard_corners(centers, dimensions):
    corners = (
        centers[0][0],
        centers[dimensions[0] - 1][0],
        centers[dimensions[0] * dimensions[1] - 1][0],
        centers[dimensions[0] * (dimensions[1] - 1)][0]
    )
    corners = tuple(tuple(corner) for corner in corners)
    return corners


if __name__ == "__main__":
    main()
