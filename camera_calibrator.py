import logging
import os
import time
import datetime

__author__ = 'Mark'

import cv2
from pattern_type import PatternType
import numpy as np
import matplotlib.pyplot as plt

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

        self.map_x = np.array([])
        self.map_y = np.array([])
        self.reverse_map_x = np.array([])
        self.reverse_map_y = np.array([])
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
        logging.debug("Calculating new camera matrix.")
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
        logging.debug("Calculating new camera matrix - Complete.")

    def generate_maps(self):
        logging.debug("Generating maps")
        logging.debug("Generating distortion map")
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(self.original_camera_matrix, self.coefficients, None,
                                                             self.new_camera_matrix, self.image_size, 5)
        logging.debug("Generating distortion map - Complete")
        self._generate_reverse_map()
        logging.debug("Generating maps - Complete.")

    def save_results(self, additional_string="", csv=True):
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d__%H_%M_%S')
        directory = "results/{}_{}/".format(timestamp, additional_string)
        if os.path.exists(directory):
            logging.error("directory {} already exists. Results not saved.")
            return
        os.makedirs(directory)

        np.save("{}dist_coeffs.npy".format(directory), self.coefficients)
        np.save("{}original_camera_matrix.npy".format(directory), self.original_camera_matrix)
        np.save("{}alpha.npy".format(directory), self.alpha)
        logging.debug("Results saved in numpy format")
        self._save_cvs_results(directory)

        logging.info("Saving results - Complete.")


    def _save_cvs_results(self, directory):
        csv_directory = "{}csv/".format(directory)
        os.makedirs(csv_directory)

        np.savetxt("{}dist_coeffs.csv".format(csv_directory), self.coefficients, delimiter=", ")
        np.savetxt("{}original_camera_matrix.csv".format(csv_directory), self.original_camera_matrix, delimiter=", ")
        np.savetxt("{}alpha.csv".format(csv_directory), [self.alpha], delimiter=", ")

        # np.savetxt("{}map_x.csv".format(csv_directory), self.map_x, delimiter=", ")
        #np.savetxt("{}map_y.csv".format(csv_directory), self.map_y, delimiter=", ")

        #Maps are originally in floats, but Integer versions are smaller
        np.savetxt("{}int_map_x.csv".format(csv_directory), np.rint(self.map_x).astype(int), delimiter=", ", fmt="%4i")
        np.savetxt("{}int_map_y.csv".format(csv_directory), np.rint(self.map_y).astype(int), delimiter=", ", fmt="%4i")
        np.savetxt("{}int_reverse_map_x.csv".format(csv_directory), np.rint(self.reverse_map_x).astype(int), delimiter=", ", fmt="%4i")
        np.savetxt("{}int_reverse_map_y.csv".format(csv_directory), np.rint(self.reverse_map_y).astype(int), delimiter=", ", fmt="%4i")

    def load_results(self, timestamp):
        logging.debug("Loading results from '{}".format(timestamp))
        directory = "results/{}/".format(timestamp)

        self.coefficients = np.load("{}dist_coeffs.npy".format(directory))
        self.original_camera_matrix = np.load("{}original_camera_matrix.npy".format(directory))
        self.alpha = np.load("{}alpha.npy".format(directory))

        self.new_camera_matrix = self.original_camera_matrix
        logging.debug("Loading results - Complete")

    def distort_point(self, point):
        return self._remap_point(point, (self.map_x, self.map_y))

    def undistort_point(self, point):
        return self._remap_point(point, (self.reverse_map_x, self.reverse_map_y))

    @staticmethod
    def _remap_point(point, maps):
        map_x, map_y = maps
        x, y = point
        return map_x[y][x], map_y[y][x]

    def _generate_reverse_map(self):
        logging.debug("Generating undistort(reverse) map")
        # Currently generates reverse map with same size
        self.reverse_map_x = np.empty_like(self.map_x)
        self.reverse_map_y = np.empty_like(self.map_y)
        self.reverse_map_x[:] = np.NAN
        self.reverse_map_y[:] = np.NAN

        for row_no, rows in enumerate(zip(self.map_x, self.map_y)):
            for col_no, elements in enumerate(zip(*rows)):
                try:
                    self.reverse_map_x[elements[1]][elements[0]] = col_no
                    self.reverse_map_y[elements[1]][elements[0]] = row_no
                except IndexError:
                    pass
        logging.debug("Initial reverse map generated, now elliminating NAN's if any")
        while np.isnan(self.reverse_map_x).any():
            temp_x = self.reverse_map_x.copy()
            temp_y = self.reverse_map_y.copy()

            #Count the number of NAN's for debugging
            number_of_NANs = np.count_nonzero(~np.isnan(temp_x))
            assert number_of_NANs == np.count_nonzero(~np.isnan(temp_y)), "map_y must contain as many NAN's as does map_x"
            logging.debug("Number of NAN-values: {}".format(number_of_NANs))

            #Iterate through the map looking for NAN values
            for row_no, row in enumerate(temp_x):
                for col_no, element in enumerate(row):
                    if np.isnan(element):
                        #NAN found,
                        #Iterate through adjacent cells and try to find value there
                        for i in (1, -1):
                            for j in (1, -1):
                                try:
                                    adjecent_x = self.reverse_map_x[row_no + i][col_no + j]
                                    adjecent_y = self.reverse_map_y[row_no + i][col_no + j]
                                except IndexError:
                                    pass
                                else:
                                    if not np.isnan(adjecent_x) and not np.isnan(adjecent_y):
                                        temp_x[row_no][col_no] = adjecent_x
                                        temp_y[row_no][col_no] = adjecent_y
            self.reverse_map_x = temp_x
            self.reverse_map_y = temp_y

        logging.debug("Generating undistort(reverse) map - Complete")

    def plot(self):
        screen_x = (0,0,self.image_size[0],self.image_size[0], 0)
        screen_y = (0,self.image_size[1],self.image_size[1], 0, 0)

        plt.figure(1)
        plt.subplot(121)
        x = self.map_x.ravel()
        y = self.map_y.ravel()
        plt.plot(x, y, 'r.', markersize=1)
        plt.plot(screen_x, screen_y, 'k-')

        plt.subplot(122)
        x = self.reverse_map_x.ravel()
        y = self.reverse_map_y.ravel()
        plt.plot(x, y, 'g.', markersize=1)
        plt.plot(screen_x, screen_y, 'k-')

        plt.show()
