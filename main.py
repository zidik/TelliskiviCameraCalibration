import cv2
from enum import Enum
import time

from pattern_type import PatternType
from pattern_finder import PatternFinder
from camera_calibrator import CameraCalibrator
from geometry import line_intersection
from drawing_helpers import draw_horizontal_line, draw_vertical_line, draw_corners
from coordinate_mapper import find_checkerboard_corners, CoordinateMapper, Corner


class Mode(Enum):
    Initial = 0,
    Calibration = 1,
    Calibrated = 2


def main():
    cap = cv2.VideoCapture(0)

    pattern_dims = (5, 7)
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
                        show_ui_taking_sample(frame)

                    if calibrator.number_of_samples > 100:
                        show_ui_calibrating(frame)
                        ret, map_x, map_y, roi = calibrator.calibrate(frame.shape)
                        print("Calibration error: {}".format(ret))
                        mode = Mode.Calibrated

                cv2.drawChessboardCorners(frame, pattern_dims, pattern_points, pattern_found)

                #Find four corners of the board and use then to calculate mapping constants
                corners = find_checkerboard_corners(pattern_points, pattern_dims)
                #TODO: This should be done shomehow differently?
                coordinate_mapper.image_dims = (frame.shape[1], frame.shape[0])
                coordinate_mapper.calculate_constants(corners)
                coordinate_mapper.draw_intersection_lines(frame, corners)
                coordinate_mapper.draw_grid(frame)
                print("Constants", coordinate_mapper.constants)
                draw_corners(frame, corners)

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
