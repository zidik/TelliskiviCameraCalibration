__author__ = 'Mark'

import cv2
import threading
import copy

from pattern_type import PatternType

class PatternFinder(threading.Thread):
    # Read only properties:
    @property
    def recognition_in_progress(self):
        return self._new_data.is_set()

    @property
    def pattern_found(self):
        return self._pattern_found

    @property
    def pattern_points(self):
        return copy.deepcopy(self._pattern_points)

    def __init__(self, pattern_type, pattern_dims):
        super().__init__()
        self.pattern_type = pattern_type
        self.pattern_dims = pattern_dims
        self._new_data = threading.Event()

        # Input
        self._frame = None

        #Output
        self._pattern_found = False
        self._pattern_points = None

        self.daemon = True

    def run(self):
        while True:
            self._new_data.wait()
            self._find_pattern()

    def start_pattern_recognition(self, frame, copy_frame=True):
        """
        Starts the pattern recognition from image.

        :param frame: input image
        :param copy_frame: whether the thread makes it's own copy of image before starting the pattern recognition.
        :raises RuntimeError: when start_pattern_recognition is called while last pattern recognition is not yet
        complete.
        """
        if self._new_data.is_set():
            raise RuntimeError("Tried to start new pattern recognition before last one was complete.")

        if copy_frame:
            self._frame = copy.deepcopy(frame)
        else:
            self._frame = frame

        #Signal thread
        self._new_data.set()

    def _find_pattern(self):
        """
        Tries to find pattern from 'self.frame'
        If found, 'self._pattern_found' will be set to "True" and 'self._pattern_points' will be filled with
        coordinates.
        If pattern is not found in the frame 'self._pattern_found' will be set to "False"
        When the recognition has ended, Event 'self._new_data' will be cleared
        """
        gray = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY)
        if self.pattern_type == PatternType.Checkerboard:
            self._pattern_found, self._pattern_points = cv2.findChessboardCorners(
                image=gray,
                patternSize=self.pattern_dims,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if self._pattern_found:
                # Improve found points' accuracy
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
                self._pattern_points = cv2.cornerSubPix(gray, self._pattern_points, (11, 11), (-1, -1), criteria)

        elif self.pattern_type == PatternType.AsymmetricCircles:
            self._pattern_found, self._pattern_points = cv2.findCirclesGrid(
                image=gray,
                patternSize=self.pattern_dims,
                flags=cv2.CALIB_CB_ASYMMETRIC_GRID  # + cv2.CALIB_CB_CLUSTERING
            )

        self._new_data.clear()
