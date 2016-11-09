import cv2
import numpy as np

from pose_estimator import PoseEstimator

class ImagePoseEstimator(PoseEstimator):

    def _estimate(self, model, marker, data):

        success, rvec, tvec = cv2.solvePnP(model.points, marker.points, data["camera"].K, data["camera"].D)
        if not success: return None

        R = cv2.Rodrigues(rvec)[0]
        RFix = np.diag([1,-1,-1])
        R = R.dot(RFix)
        rvec,_ = cv2.Rodrigues(R)

        return rvec, tvec
