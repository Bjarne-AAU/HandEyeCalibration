import cv2
import numpy as np
import aruco

from detector.aruco_detector import ArucoDetector

class ArucoPoseEstimator(object):

    def __init__(self, marker_size = 0.059, camera_params = "xtion.yaml"):
        self._camera_params = aruco.CameraParameters()
        self._camera_params.readFromXMLFile(camera_params)
        self._detector = ArucoDetector(marker_size)

    def process(self, frame):
        marker = self._detector.process(frame)
        if marker is not None:
            marker.calculateExtrinsics(marker.ssize, self._camera_params, False)
        return marker

    def draw(self, frame, marker):
        if marker is not None:
            aruco.CvDrawingUtils.draw3dAxis(frame, marker, self._camera_params)
            aruco.CvDrawingUtils.draw3dCube(frame, marker, self._camera_params)

        return frame


