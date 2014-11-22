import cv2
from enum import Enum
import logging
import threading
import time
import copy

import numpy

from pattern_type import PatternType
from pattern_finder import PatternFinder
from camera_calibrator import CameraCalibrator
from drawing_helpers import draw_vertical_line, draw_corners, draw_grid
from coordinate_mapper import find_checkerboard_corners, CoordinateMapper


class Mode(Enum):
    Initial = 0,
    Calibration = 1,
    Calibrated = 2


pattern_dims = (5, 8)
pattern_type = PatternType.Checkerboard
calibrator = CameraCalibrator(pattern_type, pattern_dims, None)
mode = Mode.Initial

crop_scale = 0.0


def crop_frame(frame, crop_corners):
    target_frame = frame[
                   crop_corners[0][1]:crop_corners[1][1],
                   crop_corners[0][0]:crop_corners[1][0]
    ]
    return target_frame


def configure_windows():
    cv2.namedWindow('distorted')
    cv2.namedWindow('undistorted')
    cv2.createTrackbar('alpha', 'undistorted', 0, 100, change_alpha)
    cv2.setTrackbarPos('alpha', 'undistorted', int(calibrator.alpha * 100))
    cv2.createTrackbar('crop', 'undistorted', 0, 100, change_crop)


def start_camera():
    logging.debug("Starting camera")
    cap = cv2.VideoCapture(cv2.CAP_XIAPI)
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    logging.debug("Camera started")
    return cap


def main():
    global mode

    logging.basicConfig(format='[%(asctime)s] [%(threadName)13s] %(levelname)7s: %(message)s', level=logging.DEBUG)

    #Start camera
    cap = start_camera()

    #Configure windows
    configure_windows()

    #Setup pattern finder
    pattern_finder = PatternFinder(pattern_type, pattern_dims)
    pattern_finder.start()

    #Setup calibration variables
    pattern_found = False
    last_calibration_sample_time = None

    #Setup coordinate mapper
    coordinate_mapper = CoordinateMapper(
        checkerboard_distance=0.20,
        checkerboard_width=0.279,
        checkerboard_height=0.159
    )

    #Setup display flags
    show_coordinate_grid = False

    while True:
        #Try to get a frame from camera
        success, distorted_frame_clean = cap.read()
        if not success:
            logging.warning("Could not retrieve frame from camera")
            continue  # Let's just try again

        #Make original frame read-only
        distorted_frame_clean.flags.writeable = False
        image_size = tuple(distorted_frame_clean.shape[0:2][::-1])
        calibrator.image_size = image_size

        #Create copy that we can draw on
        distorted_frame = copy.deepcopy(distorted_frame_clean)

        """
        #Draw a grid on distorted frame
        draw_grid(
            frame=distorted_frame,
            x_range=(0, image_size[0], image_size[0] // 10),
            y_range=(0, image_size[1], image_size[1] // 10),
            color=(0, 0, 0), thickness=1, line_type=cv2.LINE_AA
        )
        """

        undistorted_frame_clean = None
        undistorted_frame = None

        if mode == Mode.Calibrated:
            #If calibration has been loaded, show the
            undistorted_frame_clean = cv2.remap(distorted_frame_clean, calibrator.map_x, calibrator.map_y,
                                                cv2.INTER_LINEAR)
            undistorted_frame_clean.flags.writeable = False
            undistorted_frame = cv2.remap(distorted_frame, calibrator.map_x, calibrator.map_y, cv2.INTER_LINEAR)

        if not pattern_finder.recognition_in_progress:
            #Get the results of last recognition
            pattern_found = pattern_finder.pattern_found
            pattern_points = pattern_finder.pattern_points

            #Start a new recognition
            #Calculate the corners of cropped image according to crop_scale
            crop_corners = (
                (int(image_size[0]/2*crop_scale), int(image_size[1]/2*crop_scale)),
                (int(image_size[0] - image_size[0]/2*crop_scale), int(image_size[1] - image_size[1]/2*crop_scale))
            )
            if pattern_points is not None:
                pattern_points += crop_corners[0]

            #If calibration has been loaded, let's find pattern from calibrated image
            if mode == Mode.Calibrated:
                target_frame = undistorted_frame
                target_frame_clean = undistorted_frame_clean
            else:
                target_frame = distorted_frame
                target_frame_clean = distorted_frame_clean

            #Show crop on target frame with a rectangle, and crop the clean frame
            cv2.rectangle(target_frame, crop_corners[0], crop_corners[1], (255, 0, 0), 1, cv2.LINE_AA)
            cropped_frame = crop_frame(target_frame_clean, crop_corners)

            #Start a new pattern Recognition
            pattern_finder.start_pattern_recognition(cropped_frame)

        if pattern_found:
            if mode == Mode.Calibration:
                if last_calibration_sample_time is None or time.time() - last_calibration_sample_time > 0.2:
                    last_calibration_sample_time = time.time()
                    calibrator.add_sample(pattern_points)
                    show_ui_taking_sample(distorted_frame)

                if calibrator.number_of_samples > 100:
                    show_ui_calibrating(distorted_frame)
                    calibrator.calibrate()
                    calibrator.calculate_new_camera_matrix()
                    calibrator.generate_maps()
                    logging.info("Calibration Finished")
                    logging.info("Calibration error-rate: {}".format(calibrator.accuracy))
                    mode = Mode.Calibrated
                    calibrator.save_results("AutoSave")
                    continue  #Needed for initialising "undistorted_frame"

            # If view is calibrated, then draw chessboard on undistorted frame, otherwise use the original/distorted one.
            if mode == Mode.Calibrated:
                target_frame = undistorted_frame
            else:
                target_frame = distorted_frame
            cv2.drawChessboardCorners(target_frame, pattern_dims, pattern_points, pattern_found)

            # Find four corners of the board and use then to calculate mapping constants
            corners = find_checkerboard_corners(pattern_points, pattern_dims)

            # TODO: This should be done somehow differently?
            coordinate_mapper.image_dims = (distorted_frame_clean.shape[1], distorted_frame_clean.shape[0])

            coordinate_mapper.calculate_constants(corners)
            if show_coordinate_grid:
                coordinate_mapper.draw_intersection_lines(target_frame, corners)
                coordinate_mapper.draw_grid(target_frame)
                draw_corners(target_frame, corners)
                # Vertical line in the center
                draw_vertical_line(target_frame, target_frame.shape[1] // 2, (100, 0, 0), 1,
                                   cv2.LINE_AA)

            logging.info("Constants {}".format(coordinate_mapper.constants))

        # Display the distorted frame (original with additional lines)
        distorted_frame = cv2.resize(distorted_frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('distorted', distorted_frame)

        if mode == Mode.Calibrated:
            undistorted_frame = cv2.resize(undistorted_frame, (0, 0), fx=0.5, fy=0.5)

            cv2.imshow('undistorted', undistorted_frame)

            # TODO: Why does it work strange?
            #if calibrator.roi is not None:
            #    x, y, w, h = [elem // 2 for elem in calibrator.roi]
            #    if w != 0 and h != 0:
            #        cropped = undistorted_frame[y:y + h, x:x + w]
            #        cv2.imshow('undistorted_cropped', cropped)


        key_no = cv2.waitKey(30) & 0xFF
        if key_no == 255:
            # no key was pressed:
            pass
        elif key_no == ord('q'):
            logging.info("Quitting...")
            break
        elif key_no == ord('g'):
            show_coordinate_grid = not show_coordinate_grid
        elif key_no == ord('c'):
            logging.info("Calibration started")
            calibrator.clear()
            mode = Mode.Calibration
        elif key_no == ord('s'):
            calibrator.save_results("ManualSave")
            logging.info("Calibration results saved")
        elif key_no == ord('l'):
            timestamp = input("Type timestamp to load: ")
            try:
                calibrator.load_results(timestamp)
            except IOError:
                logging.exception("Could not load all calibration files.")
            else:
                mode = Mode.Calibrated
                calibrator.calculate_new_camera_matrix()
                calibrator.generate_maps()
                cv2.setTrackbarPos('alpha', 'undistorted', int(calibrator.alpha * 100))

                logging.info("Calibration results loaded")

        elif key_no == ord('p'):
            calibrator.plot()

        else:
            print("Press:\n"
                  "\t'q' to quit\n"
                  "\t'c' to start calibration\n"
                  "\t'g' to toggle grid\n"
                  "\t's' to save calibration results\n"
                  "\t'l' to load calibration results\n"
            )
    ## End of the main loop ##
    cap.release()
    cv2.destroyAllWindows()


last_change = None
target_alpha = None

def timing_function():
    global last_change
    while True:
        time.sleep(0.1)
        if last_change is not None and time.time()-last_change > 1.0:
            calibrator.calculate_new_camera_matrix(target_alpha)
            calibrator.generate_maps()
            last_change = None

timing_thread = threading.Thread(target=timing_function)
timing_thread.daemon = True
timing_thread.start()

def change_alpha(value):
    global target_alpha, last_change
    if mode != Mode.Calibrated:
        return
    target_alpha = value / 100
    last_change = time.time()





def change_crop(value):
    global crop_scale
    crop_scale = value/100


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
    cv2.imshow('distorted', frame)
    cv2.waitKey(20)


def show_ui_taking_sample(frame):
    color = (255, 255, 255)
    cv2.circle(frame, (frame.shape[1]//10,      frame.shape[0]//10),    frame.shape[0]//25, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (frame.shape[1]//10*9,    frame.shape[0]//10),    frame.shape[0]//25, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (frame.shape[1]//10,      frame.shape[0]//10*9),  frame.shape[0]//25, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (frame.shape[1]//10*9,    frame.shape[0]//10*9),  frame.shape[0]//20, color, -1, cv2.LINE_AA)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Uncaught exception from main():")
