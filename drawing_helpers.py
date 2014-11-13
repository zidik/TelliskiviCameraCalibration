__author__ = 'Mark'

import cv2

def draw_corners(frame, corners):
    count = 0
    for name, point in corners.items():
        count +=1
        cv2.circle(frame, point, radius=5, color=(0, 0, 255), thickness=2)
        cv2.putText(
            img=frame,
            text="{} {}".format(count, name),
            org=point,
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1.5,
            color=(100, 0, 200),
            thickness=1,
            lineType=cv2.LINE_AA
        )

def draw_horizontal_line(frame, y, color, thickness, line_type=0, shift=0):
    cv2.line(frame, (0, y), (frame.shape[1], y), color, thickness, line_type, shift)


def draw_vertical_line(frame, x, color, thickness, line_type=0, shift=0):
    cv2.line(frame, (x, 0), (x, frame.shape[0]), color, thickness, line_type, shift)


def draw_grid(frame, x_range, y_range, color, thickness, line_type=0, shift=0):
    for x in range(*x_range):
        draw_vertical_line(frame, x, color, thickness, line_type, shift)
    for y in range(*y_range):
        draw_horizontal_line(frame, y, color, thickness, line_type, shift)