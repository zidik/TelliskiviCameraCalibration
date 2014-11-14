import logging
import os
import time
import datetime

__author__ = 'Mark'

import cv2
from pattern_type import PatternType
import numpy as np

class CameraCalibrator:
    def __init__(self, pattern_type, pattern_dims, image_size):
        if pattern_type is not PatternType.Checkerboard:
            raise NotImplementedError("Currently implemented only for checkerboard pattern")

        self.pattern_dims = pattern_dims
        self.image_size = image_size

        #Position of calibration points in real world: just a grid on x,y dimensions (z=0)
        self._pattern_object_points = np.zeros((self.pattern_dims[0] * self.pattern_dims[1], 3), np.float32)
        self._pattern_object_points[:, :2] = np.mgrid[0:self.pattern_dims[0], 0: self.pattern_dims[1]].T.reshape(-1, 2)

        self.number_of_samples = 0
        self._calibration_object_points = []
        self._calibration_samples = []

        # Calibration results:
        self.accuracy = 0
        self.coefficients = []
        self.original_camera_matrix = []
        self.new_camera_matrix = []
        self.map_x = []
        self.map_y = []
        self.roi = []
        self.alpha = 0


    def clear(self):
        self.number_of_samples = 0
        self._calibration_object_points = []
        self._calibration_samples = []


    def add_sample(self, pattern_points):
        self._calibration_object_points.append(self._pattern_object_points)
        self._calibration_samples.append(pattern_points)
        self.number_of_samples += 1

    def calibrate(self):
        self.accuracy, self.original_camera_matrix, self.coefficients, _, _ = \
            cv2.calibrateCamera(self._calibration_object_points, self._calibration_samples, self.image_size, None, None)
        logging.info("distortion coeffs (k1,k2,p1,p2[,k3[,k4,k5,k6]]) = {}".format(self.coefficients))
        # Use the same camera_matrix
        self.roi = None
        self.new_camera_matrix = self.original_camera_matrix

    def calculate_new_camera_matrix(self, alpha=None, center_principal_point=True):
        if alpha is not None:
            self.alpha = alpha
        # Calculate new camera matrix
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.original_camera_matrix,
            self.coefficients,
            self.image_size,
            self.alpha,
            self.image_size,
            centerPrincipalPoint=center_principal_point
        )

    def remap(self):
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(self.original_camera_matrix, self.coefficients, None,
                                                             self.new_camera_matrix, self.image_size, 5)

    def save_results(self, additional_string=""):
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d__%H_%M_%S')
        directory = "results/{}_{}/".format(timestamp, additional_string)
        if os.path.exists(directory):
            logging.error("directory {} already exists")
            return
        os.makedirs(directory)
        np.save("{}dist_coeffs.npy".format(directory), self.coefficients)
        np.save("{}original_camera_matrix.npy".format(directory), self.original_camera_matrix)
        np.save("{}map_x.npy".format(directory), self.map_x)
        np.save("{}map_y.npy".format(directory), self.map_y)

        np.save("{}alpha.npy".format(directory), self.alpha)

        csv_directory = "{}csv/".format(directory)
        os.makedirs(csv_directory)
        np.savetxt("{}dist_coeffs.csv".format(csv_directory), self.coefficients, delimiter=", ")
        np.savetxt("{}original_camera_matrix.csv".format(csv_directory), self.original_camera_matrix, delimiter=", ")

        # np.savetxt("{}map_x.csv".format(csv_directory), self.map_x, delimiter=", ")
        #np.savetxt("{}map_y.csv".format(csv_directory), self.map_y, delimiter=", ")
        np.savetxt("{}int_map_x.csv".format(csv_directory), np.rint(self.map_x).astype(int), delimiter=", ", fmt="%4i")
        np.savetxt("{}int_map_y.csv".format(csv_directory), np.rint(self.map_y).astype(int), delimiter=", ", fmt="%4i")

        np.savetxt("{}alpha.csv".format(csv_directory), [self.alpha], delimiter=", ")

    def load_results(self, timestamp):
        directory = "results/{}/".format(timestamp)
        self.coefficients = np.load("{}dist_coeffs.npy".format(directory))
        self.original_camera_matrix = np.load("{}original_camera_matrix.npy".format(directory))
        self.alpha = np.load("{}alpha.npy".format(directory))
        self.new_camera_matrix = self.original_camera_matrix
        self.remap()
        #self.map_x = np.load("{}map_x.npy".format(directory))
        #self.map_y = np.load("{}map_y.npy".format(directory))
