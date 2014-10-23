__author__ = 'Mark'

import cv2

def draw_numbered_points(frame, points):
    count = 0
    for point in points:
        count += 1
        cv2.circle(frame, point, radius=5, color=(0, 0, 255), thickness=2)
        cv2.putText(frame, "{}".format(count), point, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)


def draw_horizontal_line(frame, y, color, thickness, line_type=0, shift=0):
    cv2.line(frame, (0, y), (frame.shape[1], y), color, thickness, line_type, shift)


def draw_vertical_line(frame, x, color, thickness, line_type=0, shift=0):
    cv2.line(frame, (x, 0), (x, frame.shape[0]), color, thickness, line_type, shift)