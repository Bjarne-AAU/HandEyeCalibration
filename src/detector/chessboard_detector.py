import cv2
import numpy as np

from detector import Marker
from detector import Detector


class ChessboardMarker(Marker):

    @classmethod
    def create(self, width, height, size):
        bwidth = (width-1)*size/2.0
        bheight = (height-1)*size/2.0
        points = np.mgrid[bwidth:-bwidth:np.complex(width), -bheight:bheight:np.complex(height)].T.reshape(-1,2, order='F')
        points = np.insert(points, 2, values=0, axis=1)
        return ChessboardMarker(points, (height, width))

    def __init__(self, points, tiles):
        super(ChessboardMarker, self).__init__(points)
        self._tiles = tiles

    @property
    def tiles(self):
        return self._tiles

    @property
    def shape(self):
        return self._points[[-self.tiles[0],0,self._tiles[0]-1,-1]]

    @property
    def box(self):
        return self.shape


class ChessboardDetector(Detector):

    CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def __init__(self, width, height, size):
        model = ChessboardMarker.create(width, height, size)
        super(ChessboardDetector, self).__init__(model)

    def _draw(self, frame, marker):
        cv2.drawChessboardCorners(frame, marker.tiles, marker.points.astype(np.float32), True)
        return frame

    def _detect(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        [success, points] = cv2.findChessboardCorners(frame, self.model.tiles, \
                                                      flags=cv2.CALIB_CB_FAST_CHECK | \
                                                            cv2.CALIB_CB_ADAPTIVE_THRESH | \
                                                            cv2.CALIB_CB_NORMALIZE_IMAGE | \
                                                            cv2.CALIB_CB_FILTER_QUADS)

        if not success: return None

        cv2.cornerSubPix(frame, points, (11,11), (-1,-1), self.CRITERIA)

        points = points.reshape(-1,2)
        if points[0,1] > points[-1,1]:
            points = points[::-1]

        return ChessboardMarker(points, self.model.tiles)


