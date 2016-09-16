import cv2
import numpy as np
import aruco

# class ArucoModel(object):

#     def __init__(self, size=0.03, id=None):
#         self._width = size
#         self._height = size
#         self._id = id

#     @property
#     def size(self):
#         return (self._width, self._height)

#     @property
#     def width(self):
#         return self._width

#     @property
#     def height(self):
#         return self._height

#     # @property
#     # def points(self):
#     #     return self._grid



class ArucoDetector(object):

    def __init__(self, marker_size = 0.059, marker_id=None):
        self._marker_size = marker_size
        self._marker_id = marker_id
        self._detector = aruco.MarkerDetector()
        self._detector.setThresholdMethod(aruco.MarkerDetector.NONE)
        self._detector.setCornerRefinementMethod(aruco.MarkerDetector.LINES)

        # self._detector.setThresholdMethod(aruco.MarkerDetector.ADPT_THRES)
        # self._detector.setThresholdMethod(aruco.MarkerDetector.FIXED_THRES)
        # self._detector.setThresholdParams(20, 50)
        # self._detector.enableLockedCornersMethod(True)
        # self._detector.enableErosion(True)
        # self._detector.setCornerRefinementMethod(aruco.MarkerDetector.LINES)
        # self._detector.setCornerRefinementMethod(aruco.MarkerDetector.SUBPIX)
        # self._detector.setThresholdParams(0, 150)

    def process(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -4)
        marker = self._detector.detect(frame, aruco.CameraParameters())
        for m in marker:
            m.ssize = self._marker_size
        if len(marker) == 0: return None
        if self._marker_id is None: return marker[0]
        marker = [marker for m in marker if m.id == self._id]
        return marker[0] if len(marker) > 0 else None

    def draw(self, frame, marker):
        if marker is not None:
            marker.draw(frame, np.array([0, 0, 255]), 2)

        return frame


