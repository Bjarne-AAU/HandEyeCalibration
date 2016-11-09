from tools.opencv3 import enable_cv3
enable_cv3("/usr/local/opencv3/lib")

import cv2
import numpy as np

from detector import Marker
from detector import Detector

if hasattr(cv2, 'aruco'):

    class ArucoMarker(Marker):

        @classmethod
        def create(self, size, id=None):
            s = size/2.0
            points = np.zeros((4,3))
            points[0] = [-s, -s, 0]
            points[1] = [ s, -s, 0]
            points[2] = [ s,  s, 0]
            points[3] = [-s,  s, 0]
            return ArucoMarker(points, id)

        def __init__(self, points, id):
            super(ArucoMarker, self).__init__(points)
            self._id = id if id else None

        @property
        def id(self):
            return self._id


    class ArucoDetector(Detector):

        DICTIONARY = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)

        def __init__(self, size, id=None):
            model = ArucoMarker.create(size, id)
            super(ArucoDetector, self).__init__(model)

            self._params = cv2.aruco.DetectorParameters_create()
            self._params.minMarkerPerimeterRate = 0.1
            self._params.doCornerRefinement = True
            # print("adaptiveThreshConstant = " + str(self._params.adaptiveThreshConstant))
            # print("adaptiveThreshWinSizeMax = " + str(self._params.adaptiveThreshWinSizeMax))
            # print("adaptiveThreshWinSizeMin = " + str(self._params.adaptiveThreshWinSizeMin))
            # print("adaptiveThreshWinSizeStep = " + str(self._params.adaptiveThreshWinSizeStep))
            # print("cornerRefinementMaxIterations = " + str(self._params.cornerRefinementMaxIterations))
            # print("cornerRefinementMinAccuracy = " + str(self._params.cornerRefinementMinAccuracy))
            # print("cornerRefinementWinSize = " + str(self._params.cornerRefinementWinSize))
            # print("doCornerRefinement = " + str(self._params.doCornerRefinement))
            # print("errorCorrectionRate = " + str(self._params.errorCorrectionRate))
            # print("markerBorderBits = " + str(self._params.markerBorderBits))
            # print("maxErroneousBitsInBorderRate = " + str(self._params.maxErroneousBitsInBorderRate))
            # print("maxMarkerPerimeterRate = " + str(self._params.maxMarkerPerimeterRate))
            # print("minCornerDistanceRate = " + str(self._params.minCornerDistanceRate))
            # print("minDistanceToBorder = " + str(self._params.minDistanceToBorder))
            # print("minMarkerDistanceRate = " + str(self._params.minMarkerDistanceRate))
            # print("minMarkerPerimeterRate = " + str(self._params.minMarkerPerimeterRate))
            # print("minOtsuStdDev = " + str(self._params.minOtsuStdDev))
            # print("perspectiveRemoveIgnoredMarginPerCell = " + str(self._params.perspectiveRemoveIgnoredMarginPerCell))
            # print("perspectiveRemovePixelPerCell = " + str(self._params.perspectiveRemovePixelPerCell))
            # print("polygonalApproxAccuracyRate = " + str(self._params.polygonalApproxAccuracyRate))


        def _draw(self, frame, marker):
            frame = cv2.aruco.drawDetectedMarkers( frame, [np.array([marker.points])], np.array([marker.id]) )
            return frame

        def _detect(self, frame):
            points, ids, _ = cv2.aruco.detectMarkers(frame, ArucoDetector.DICTIONARY, parameters=self._params)

            idx = 0
            if ids is None: return None
            elif self.model.id is None: idx = 0
            elif self.model.id in ids:  idx = list(ids).index(self.model.id)
            else: return None

            return ArucoMarker(np.squeeze(points[idx]), ids[idx])
