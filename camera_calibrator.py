__author__ = 'Mark'

import cv2
from pattern_type import PatternType
import numpy as np

class CameraCalibrator:
    def __init__(self, pattern_type, pattern_dims):
        if pattern_type is not PatternType.Checkerboard:
            raise NotImplementedError("Currently implemented only for checkerboard pattern")

        self.pattern_dims = pattern_dims

        #Position of calibration points in real world: just a grid on x,y dimensions (z=0)
        self._pattern_object_points = np.zeros((self.pattern_dims[0] * self.pattern_dims[1], 3), np.float32)
        self._pattern_object_points[:, :2] = np.mgrid[0:self.pattern_dims[0], 0: self.pattern_dims[1]].T.reshape(-1, 2)

        self.number_of_samples = 0
        self._calibration_object_points = []
        self._calibration_samples = []

    def clear(self):
        self.number_of_samples = 0
        self._calibration_object_points = []
        self._calibration_samples = []


    def add_sample(self, pattern_points):
        self._calibration_object_points.append(self._pattern_object_points)
        self._calibration_samples.append(pattern_points)
        self.number_of_samples += 1

    def calibrate(self, input_image_shape):
        h, w = input_image_shape[0:2]
        ret, camera_matrix, dist, rvecs, tvecs = \
            cv2.calibrateCamera(self._calibration_object_points, self._calibration_samples, (w, h), None, None)
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (w, h), 0, (w, h))

        #map_x, map_y = cv2.initUndistortRectifyMap(camera_matrix, dist, None, new_camera_matrix, (w, h), 5)

        map_x, map_y = cv2.initUndistortRectifyMap(camera_matrix, dist, None, new_camera_matrix, (w, h), 5)


        print("Width, Height = {}, {}".format(w, h))
        print("Original Camera Matrix = \n{}".format(camera_matrix))
        matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (w, h), 0, (w, h))
        print("0 Camera Matrix = \n{} \nroi = {}".format(matrix, roi))
        matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (w, h), 1, (w, h))
        print("1 Camera Matrix = \n{} \nroi = {}".format(matrix, roi))
        print("distortion coeffs (k1,k2,p1,p2[,k3[,k4,k5,k6]]) = {}".format(dist))
        return ret, map_x, map_y, roi