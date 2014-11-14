import cv2
from enum import Enum
import logging
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


# pattern_dims = (5, 7)
pattern_dims = (8, 5)
pattern_type = PatternType.Checkerboard
calibrator = CameraCalibrator(pattern_type, pattern_dims, None)
mode = Mode.Initial

crop_scale = 0.0


def main():
    global mode

    logging.basicConfig(format='[%(asctime)s] [%(threadName)13s] %(levelname)7s: %(message)s', level=logging.DEBUG)

    logging.debug("Starting camera")
    # cap = cv2.VideoCapture(cv2.CAP_XIAPI)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Could not open camera")
        return
    logging.info("Camera started")

    cv2.namedWindow('distorted')
    cv2.namedWindow('undistorted')
    cv2.createTrackbar('alpha', 'undistorted', 0, 100, change_alpha)
    cv2.setTrackbarPos('alpha', 'undistorted', int(calibrator.alpha * 100))
    cv2.createTrackbar('crop', 'undistorted', 0, 100, change_crop)

    pattern_finder = PatternFinder(pattern_type, pattern_dims)
    pattern_finder.start()

    pattern_found = False
    last_calibration_sample_time = None

    coordinate_mapper = CoordinateMapper(
        checkerboard_distance=0.20,
        checkerboard_width=0.16,
        checkerboard_height=0.28
    )
    show_coordinate_grid = False

    while True:
        success, clean_distorted_frame = cap.read()
        if not success:
            logging.warning("Could not retrieve frame from camera")
            continue  # Let's just try again
        clean_distorted_frame.flags.writeable = False
        image_size = tuple(clean_distorted_frame.shape[0:2][::-1])
        calibrator.image_size = image_size

        # we will keep original intact
        distorted_frame = copy.deepcopy(clean_distorted_frame)
        draw_grid(
            frame=distorted_frame,
            x_range=(0, image_size[0], image_size[0] // 10),
            y_range=(0, image_size[1], image_size[1] // 10),
            color=(0, 0, 0), thickness=1, line_type=cv2.LINE_AA
        )

        crop_corners = (
            (int(image_size[0]/2*crop_scale), int(image_size[1]/2*crop_scale)),
            (int(image_size[0] - image_size[0]/2*crop_scale), int(image_size[1] - image_size[1]/2*crop_scale))
        )

        clean_undistorted_frame = None
        undistorted_frame = None
        if mode == Mode.Calibrated:
            clean_undistorted_frame = cv2.remap(clean_distorted_frame, calibrator.map_x, calibrator.map_y,
                                                cv2.INTER_LINEAR)
            clean_undistorted_frame.flags.writeable = False
            undistorted_frame = cv2.remap(distorted_frame, calibrator.map_x, calibrator.map_y, cv2.INTER_LINEAR)

        if not pattern_finder.recognition_in_progress:
            #Get results and start a next recognition
            pattern_found = pattern_finder.pattern_found
            pattern_points = pattern_finder.pattern_points
            if pattern_points is not None:
                pattern_points += crop_corners[0]

            if mode == Mode.Calibrated:
                cv2.rectangle(undistorted_frame, crop_corners[0], crop_corners[1], (255, 0, 0), 1, cv2.LINE_AA)
                target_frame = clean_undistorted_frame[
                               crop_corners[0][1]:crop_corners[1][1],
                               crop_corners[0][0]:crop_corners[1][0]
                ]
                cv2.imshow("test", target_frame)
            else:
                target_frame = clean_distorted_frame
            pattern_finder.start_pattern_recognition(target_frame)

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
                    calibrator.remap()
                    logging.info("Calibration Finished")
                    logging.info("Calibration error-rate: {}".format(calibrator.accuracy))
                    mode = Mode.Calibrated
                    calibrator.save_results("AutoSave")
                    continue  #Needed for initialising "undistorted_frame"

            # If view is calibrated, then draw chessboard on undistorted frame, otherwise use the original/distorted one.
            if mode == Mode.Calibrated:
                cv2.drawChessboardCorners(undistorted_frame, pattern_dims, pattern_points, pattern_found)

                pattern_points_translated = numpy.zeros_like(pattern_points)
                for index, element in enumerate(pattern_points):
                    x = element[0][0]
                    y = element[0][1]
                    pattern_points_translated[index][0][0] = calibrator.map_x[int(y)][int(x)]
                    pattern_points_translated[index][0][1] = calibrator.map_y[int(y)][int(x)]

                cv2.drawChessboardCorners(undistorted_frame, pattern_dims, pattern_points_translated, pattern_found)
            else:
                cv2.drawChessboardCorners(distorted_frame, pattern_dims, pattern_points, pattern_found)



            # Find four corners of the board and use then to calculate mapping constants
            corners = find_checkerboard_corners(pattern_points, pattern_dims)

            # TODO: This should be done somehow differently?
            coordinate_mapper.image_dims = (clean_distorted_frame.shape[1], clean_distorted_frame.shape[0])

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
        # clean_distorted_frame = cv2.resize(clean_distorted_frame, (0, 0), fx=0.5, fy=0.5)
        #cv2.imshow('original', clean_distorted_frame)

        # Display the distorted frame (original with additional lines
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
                cv2.setTrackbarPos('alpha', 'undistorted', int(calibrator.alpha * 100))
                calibrator.calculate_new_camera_matrix()
                calibrator.remap()

                logging.info("Calibration results loaded")

            # TODO GO ON HERE!
            reverse_maps = generate_reverse_map(calibrator.map_x, calibrator.map_y)
            test = 50, 100
            print("test:{}".format(test))
            mapped = remap_point(test, (calibrator.map_x, calibrator.map_y))
            print("map:{}".format(mapped))
            reverse_mapped = remap_point(mapped, reverse_maps)
            print("reversemap:{}".format(reverse_mapped))



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


def change_alpha(value):
    if mode != Mode.Calibrated:
        return
    alpha = value / 100
    calibrator.calculate_new_camera_matrix(alpha)
    calibrator.remap()


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


def remap_point(point, maps):
    map_x, map_y = maps
    x, y = point
    return map_x[y][x], map_y[y][x]


def generate_reverse_map(map_x, map_y):
    # Currently generates reverse map with same size
    reverse_mapx = numpy.empty_like(map_x)
    reverse_mapy = numpy.empty_like(map_y)

    for row_no, rows in enumerate(zip(map_x, map_y)):
        for col_no, elements in enumerate(zip(*rows)):
            try:
                reverse_mapx[elements[1]][elements[0]] = col_no
                reverse_mapy[elements[1]][elements[0]] = row_no
            except IndexError:
                pass

    while numpy.isnan(reverse_mapx).any():
        print("Number of NAN-values: {}".format(numpy.count_nonzero(~numpy.isnan(reverse_mapx))))
        for row_no, row in enumerate(reverse_mapx):
            for col_no, element in enumerate(row):
                if numpy.isnan(element):
                    for i in (1, -1):
                        for j in (1, -1):
                            try:
                                reverse_mapx[row_no][col_no] = reverse_mapx[row_no + i][col_no + j]
                                reverse_mapx[row_no][col_no] = reverse_mapx[row_no + i][col_no + j]
                            except IndexError:
                                pass
    print("Done!")
    return reverse_mapx, reverse_mapy


if __name__ == "__main__":
    main()
