__author__ = 'Mark'

from enum import Enum
import cv2
from geometry import line_intersection
from drawing_helpers import draw_horizontal_line, draw_vertical_line

class Corner(Enum):
    BottomLeft = 0,
    BottomRight = 1,
    TopLeft = 2,
    TopRight = 3

class CoordinateMapper:
    """

    m_b = (distA*pVertA - distB*pVertB)/(pVertA - pVertB);
    m_a = (distA - m_b)*pVertA;

    float pxWidthA = (detector->corner(CheckerboardDetector::L_BOTTOMLEFT).y() - detector->corner(CheckerboardDetector::L_BOTTOMRIGHT).y());
    m_c = CHECKERBOARD_PATTERN_WIDTH*pVertA/pxWidthA;
    """
    @property
    def horizon(self):
        return self.infinity_intersection[1]

    @property
    def constants(self):
        return self.const_a, self.const_b, self.const_c, self.horizon

    def __init__(self, checkerboard_distance, checkerboard_width, checkerboard_height):
        self.checkerboard_distance = checkerboard_distance
        self.checkerboard_width = checkerboard_width
        self.checkerboard_height = checkerboard_height

        self.image_dims = (0, 0)

        self.const_a = 0
        self.const_b = 0
        self.const_c = 0
        self.infinity_intersection = (0, 0)

    def calculate_constants(self, corners):
        #Find horizon:
        self.infinity_intersection = line_intersection(
            (corners[Corner.BottomLeft], corners[Corner.TopLeft]),
            (corners[Corner.BottomRight], corners[Corner.TopRight]),
        )
        #print("Intersection", self.infinity_intersection)

        dist_bottom = self.checkerboard_distance
        dist_top = self.checkerboard_distance + self.checkerboard_height

        #print("Dist_bottom", dist_bottom)
        #print("Dist_top", dist_top)

        vert_bottom_px = corners[Corner.BottomLeft][1] - self.horizon
        vert_top_px = corners[Corner.TopLeft][1] - self.horizon

        print("vert_bottom_px", vert_bottom_px)
        print("vert_top_px", vert_top_px)

        self.const_b = (dist_bottom*vert_bottom_px - dist_top*vert_top_px)/(vert_bottom_px - vert_top_px)
        self.const_a = (dist_bottom - self.const_b)*vert_bottom_px

        width_bottom_px = abs(corners[Corner.BottomRight][0] - corners[Corner.BottomLeft][0])
        print("width_bottom_px", width_bottom_px)
        print("checkerboard_width", self.checkerboard_width)
        self.const_c = self.checkerboard_width*vert_bottom_px/width_bottom_px

    def to_world(self, x_px, y_px):
        vert_px = y_px - self.horizon
        right_px = x_px - self.image_dims[0]/2

        y_world = self.const_b + self.const_a / vert_px
        x_world = self.const_c * right_px / vert_px

        return x_world, y_world

    def to_camera(self, x_world, y_world):
        vert_px = self.const_a / (y_world - self.const_b)
        right_px = x_world * vert_px / self.const_c
        y_px = vert_px + self.horizon
        x_px = right_px + self.image_dims[0] / 2
        return x_px, y_px

    # Debug by painting a square of points -1m..1m on right, 0.4m..5m on forward, with 100mm interval.
    def draw_grid(self, frame):
        """
        Draw markers in grid
        :param frame: frame to draw on
        """
        def draw_marker(frame, position):
            """
            Draw one parker on specified position
            :param frame: frame to draw on
            :param position: position of the marker on the frame
            """
            cv2.circle(
                img=frame,
                center=position,
                radius=3,
                color=(0, 0, 0),
                thickness=-1,
                lineType=cv2.LINE_AA
            )
            cv2.circle(
                img=frame,
                center=position,
                radius=1,
                color=(255, 255, 255),
                thickness=-1,
                lineType=cv2.LINE_AA
            )

        for x in [val * 0.001 for val in range(-1000, 1000, 100)]:
            for y in [val * 0.001 for val in range(0, 1000, 100)]:
                draw_marker(frame, tuple(int(val) for val in self.to_camera(x, y)))

    def draw_intersection_lines(self, frame, corners):
        #Draw lines to intersection (horizon)
        for corner in corners.values():
            cv2.line(frame, corner, self.infinity_intersection, (150, 0, 0), 1, cv2.LINE_AA)
        draw_horizontal_line(frame, self.horizon, (200, 0, 0), 2, cv2.LINE_AA)
        draw_vertical_line(frame, self.infinity_intersection[0], (200, 0, 0), 1, cv2.LINE_AA)






def find_checkerboard_corners(centers, dimensions):
    corners = {
        Corner.TopLeft:     centers[0][0],
        Corner.BottomLeft:  centers[dimensions[0] - 1][0],
        Corner.BottomRight: centers[dimensions[0] * dimensions[1] - 1][0],
        Corner.TopRight:    centers[dimensions[0] * (dimensions[1] - 1)][0]
    }
    #Convert corner values to tuples - it is easier to feed them to OpenCV afterwards.
    for corner in corners:
        corners[corner] = tuple(corners[corner])
    return corners


def selftest():
    coordinate_mapper = CoordinateMapper(0.33, 0.41, 0.21)
    coordinate_mapper.calculate_constants([(), (), (), ()], 200)
if __name__ == "__main__":
    selftest()


