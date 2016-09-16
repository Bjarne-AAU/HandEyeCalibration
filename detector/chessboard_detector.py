import cv2
import numpy as np
import aruco

import tools.tools as tools

class ChessboardModel(object):

    def __init__(self, width=7, height=5, size=0.03):
        self._width = width
        self._height = height
        self._tile_size = size

        board_width = width*size
        board_height = height*size
        self._grid = np.zeros((width*height, 3), np.float32)
        self._grid[:,0:2] = np.mgrid[0:board_width:size, 0:board_height:size].T.reshape(-1,2)
        self._grid -= [board_width/2.0, board_height/2.0, 0]
        self._grid += [size/2.0, size/2.0, 0]

    @property
    def size(self):
        return (self._width, self._height)

    @property
    def tiles(self):
        return self._width * self._height

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def tile_size(self):
        return self._tile_size

    @property
    def board_width(self):
        return self._width * self._tile_size

    @property
    def board_height(self):
        return self._height * self._tile_size

    @property
    def points(self):
        return self._grid


class ChessboardDetector(object):

    CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def __init__(self, chessboard, camera_params="camera.yaml"):
        self._model = chessboard
        self._camera_params = aruco.CameraParameters()
        self._camera_params.readFromXMLFile(camera_params)

    @property
    def model(self):
        return self._model

    def process(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        [success, corners] = cv2.findChessboardCorners(frame, self._model.size, \
                                                       flags=cv2.CALIB_CB_ADAPTIVE_THRESH | \
                                                             cv2.CALIB_CB_NORMALIZE_IMAGE | \
                                                             cv2.CALIB_CB_FILTER_QUADS)
        if corners is None: return None
        if success:
            cv2.cornerSubPix(frame, corners, (11,11), (-1,-1), self.CRITERIA)

        corners = np.reshape(corners, (-1,2))
        if corners[0,1] > corners[-1,1]:
            corners = corners[::-1]

        return corners

    def draw(self, frame, corners):
        if corners is not None:
            success = len(corners) == self._model.tiles
            cv2.drawChessboardCorners(frame, self._model.size, corners, success)
        return frame


