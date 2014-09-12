import numpy as np
import cv2

from enum import Enum


def horizontal_line(frame, y, color, thickness, lineType=0, shift=0):
    cv2.line(frame, (0, y), (frame.shape[1], y), color, thickness, lineType, shift)

def find_corners(centers, dimensions):
    corners = (
        centers[0][0],
        centers[dimensions[0]-1][0],
        centers[dimensions[0]*dimensions[1]-1][0],
        centers[dimensions[0]*(dimensions[1]-1)][0]
    )
    corners = tuple(tuple(corner) for corner in corners)
    return corners



def line_intersection(line1, line2):
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(x_diff, y_diff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div
    return x, y



class Pattern(Enum):
    chessboard = 1
    circleboard = 2

current_pattern = Pattern.chessboard

cap = cv2.VideoCapture(0)


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
chessboard_dims = (5,7)
circlesboard_dims = (4, 11)


while(True):
    
    # Capture frame-by-frame
    success, frame = cap.read()
    if not success:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dims = None
    samples = []
    pattern_found = False
    if current_pattern == Pattern.chessboard:
        dims = chessboard_dims
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        pattern_found, corners = cv2.findChessboardCorners(gray, dims, flags=flags)
        if pattern_found:
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            cv2.drawChessboardCorners(frame, dims, corners, pattern_found)
            samples = find_corners(corners, dims)
            
    elif current_pattern == Pattern.circleboard:
        dims = circlesboard_dims
        flags = cv2.CALIB_CB_ASYMMETRIC_GRID #+ cv2.CALIB_CB_CLUSTERING
        pattern_found, centers = cv2.findCirclesGrid(gray, dims, flags=flags)
        if pattern_found:
            cv2.drawChessboardCorners(frame, dims, centers, pattern_found)
            samples = find_corners(centers, dims)

    else:
        raise ValueError("Unknown pattern \"{}\"".format(current_pattern))

    if pattern_found:
        count = 0
        intersection = line_intersection(samples[0:2], samples[2:4])
        horizontal_line(frame, intersection[1], (255, 0, 0), 2, cv2.LINE_AA)

        for corner in samples:
            count += 1
            cv2.circle(frame, corner, radius=5, color=(0, 0, 255), thickness=3)
            cv2.putText(frame, "{}".format(count), corner, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1, (255,255,255),2, cv2.LINE_AA)
            cv2.line(frame, corner, intersection, (255, 0, 0), 1, cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


