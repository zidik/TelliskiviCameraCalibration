import cv2
from enum import Enum
import logging
import time
import copy

from pattern_type import PatternType
from pattern_finder import PatternFinder
from camera_calibrator import CameraCalibrator
from drawing_helpers import draw_vertical_line, draw_corners
from coordinate_mapper import find_checkerboard_corners, CoordinateMapper


class Mode(Enum):
    Initial = 0,
    Calibration = 1,
    Calibrated = 2


def main():
    logging.basicConfig(format='[%(asctime)s] [%(threadName)13s] %(levelname)7s: %(message)s', level=logging.DEBUG)

    logging.debug("Starting camera")
    cap = cv2.VideoCapture(cv2.CAP_XIAPI)
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Could not open camera")
        return
    logging.info("Camera started")


    #pattern_dims = (5, 7)
    pattern_dims = (8, 5)

    pattern_type = PatternType.Checkerboard
    pattern_finder = PatternFinder(pattern_type, pattern_dims)
    pattern_finder.start()

    calibrator = CameraCalibrator(pattern_type, pattern_dims)
    last_calibration_sample = None
    mode = Mode.Initial

    coordinate_mapper = CoordinateMapper(
        checkerboard_distance=0.40,
        checkerboard_width=0.18,
        checkerboard_height=0.12
    )

    map_x, map_y, roi = None, None, None
    pattern_found = False
    show_coordinate_grid = False

    while True:
        success, original_frame = cap.read()
        if not success:
            logging.warning("Could not retrieve frame from camera")
            continue  # Let's just try again

        # we will keep original intact
        distorted_frame = copy.deepcopy(original_frame)

        if mode == Mode.Calibrated:
            undistorted_frame = cv2.remap(distorted_frame, map_x, map_y, cv2.INTER_LINEAR)
        else:
            undistorted_frame = None

        if not pattern_finder.recognition_in_progress:
            #Get results and start a next recognition
            pattern_found = pattern_finder.pattern_found
            pattern_points = pattern_finder.pattern_points
            pattern_finder.start_pattern_recognition(distorted_frame)

        if pattern_found:
            if mode == Mode.Calibration:
                if last_calibration_sample is None or time.time() - last_calibration_sample > 0.2:
                    last_calibration_sample = time.time()
                    calibrator.add_sample(pattern_points)
                    show_ui_taking_sample(distorted_frame)

                if calibrator.number_of_samples > 100:
                    show_ui_calibrating(distorted_frame)
                    ret, map_x, map_y, roi = calibrator.calibrate(distorted_frame.shape)
                    logging.info("Calibration error-rate: {}".format(ret))
                    mode = Mode.Calibrated
                    continue  #Needed for initialising "undistorted_frame"

            # If view is calibrated, then draw chessboard on undistorted frame, otherwise use the original/distorted one.
            if mode == Mode.Calibrated:
                chessboard_target_frame = undistorted_frame
            else:
                chessboard_target_frame = distorted_frame

            cv2.drawChessboardCorners(chessboard_target_frame, pattern_dims, pattern_points, pattern_found)

            # Find four corners of the board and use then to calculate mapping constants
            corners = find_checkerboard_corners(pattern_points, pattern_dims)

            # TODO: This should be done somehow differently?
            coordinate_mapper.image_dims = (distorted_frame.shape[1], distorted_frame.shape[0])

            coordinate_mapper.calculate_constants(corners)
            if show_coordinate_grid:
                coordinate_mapper.draw_intersection_lines(chessboard_target_frame, corners)
                coordinate_mapper.draw_grid(chessboard_target_frame)
                draw_corners(chessboard_target_frame, corners)
                # Vertical line in the center
                draw_vertical_line(chessboard_target_frame, chessboard_target_frame.shape[1] // 2, (100, 0, 0), 1,
                                   cv2.LINE_AA)

            logging.info("Constants {}".format(coordinate_mapper.constants))




        # Display the original frame
        original_frame = cv2.resize(original_frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('original', original_frame)

        # Display the distorted frame (original with additional lines
        distorted_frame = cv2.resize(distorted_frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('distorted', distorted_frame)

        if mode == Mode.Calibrated:
            undistorted_frame = cv2.resize(undistorted_frame, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('undistorted', undistorted_frame)

            if roi is not None:
                x, y, w, h = [elem / 2 for elem in roi]
                cropped = undistorted_frame[y:y + h, x:x + w]
                cv2.imshow('undistorted_cropped', cropped)

        key_no = cv2.waitKey(30) & 0xFF
        if key_no == ord('q'):
            logging.info("Quitting...")
            break
        if key_no == ord('g'):
            show_coordinate_grid = not show_coordinate_grid
        if key_no == ord('c'):
            logging.info("Calibration started")
            calibrator.clear()
            mode = Mode.Calibration
    #End (while True)
    cap.release()
    cv2.destroyAllWindows()


def show_ui_calibrating(frame):
    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], frame.shape[0]),
        (255, 255, 255),
        -1
    )
    cv2.putText(frame, "CALIBRATING...", (frame.shape[1]//3, frame.shape[0]//2),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2, cv2.LINE_AA)
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('frame', frame)
    cv2.waitKey(20)


def show_ui_taking_sample(frame):
    color = (255, 255, 255)
    cv2.circle(frame, (frame.shape[1]//10,      frame.shape[0]//10),    frame.shape[0]//25, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (frame.shape[1]//10*9,    frame.shape[0]//10),    frame.shape[0]//25, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (frame.shape[1]//10,      frame.shape[0]//10*9),  frame.shape[0]//25, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (frame.shape[1]//10*9,    frame.shape[0]//10*9),  frame.shape[0]//20, color, -1, cv2.LINE_AA)


if __name__ == "__main__":
    main()
