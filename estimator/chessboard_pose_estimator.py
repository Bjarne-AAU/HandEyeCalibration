import cv2
import numpy as np
import aruco

from detector.chessboard_detector import ChessboardDetector

import ros.conversions as convert

class ChessboardMarker(object):

    def __init__(self, rvec, tvec, size):
        self.Rvec = rvec
        self.Tvec = tvec
        self.size = size
        self.id = 0


class ChessboardPoseEstimator(object):

    CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def __init__(self, chessboard, camera_params="xtion.yaml"):
        self._camera_params = aruco.CameraParameters()
        self._camera_params.readFromXMLFile(camera_params)
        self._detector = ChessboardDetector(chessboard)

    def process(self, frame):
        model = self._detector.model
        corners = self._detector.process(frame)
        # self._detector.draw(frame, corners)
        if corners is None: return None
        if corners.shape[0] != model.tiles: return None

        cam_mat = self._camera_params.CameraMatrix
        cam_dist = self._camera_params.Distorsion
        (success, rvec, tvec) = cv2.solvePnP(model.points, corners, cam_mat, cam_dist)
        if not success: return None

        T = convert.axis2matrix(rvec, tvec)
        Rot = np.diag([1,-1,-1, 1])
        T = T.dot(Rot)
        (rvec, tvec) = convert.matrix2axis(T)

        size = max(model.board_width-model.tile_size, model.board_height-model.tile_size)
        marker = ChessboardMarker(rvec, tvec, size)

        return marker


    def draw(self, frame, marker):
        if marker is not None:
            m = aruco.Marker()
            m.Rvec = marker.Rvec
            m.Tvec = marker.Tvec
            m.ssize = marker.size
            aruco.CvDrawingUtils.draw3dAxis(frame, m, self._camera_params)
            aruco.CvDrawingUtils.draw3dCube(frame, m, self._camera_params)
        return frame


