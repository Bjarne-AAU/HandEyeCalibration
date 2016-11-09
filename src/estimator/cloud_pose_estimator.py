import cv2
import numpy as np
import scipy.linalg as linalg

from pose_estimator import PoseEstimator

class CloudPoseEstimator(PoseEstimator):

    def _estimate(self, model, marker, data):

        AXIS_STRIPE = 5
        cloud = data["cloud"]
        width = data["camera"].width
        height = data["camera"].height

        roi = np.round(marker.shape).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [roi], 255)
        marker_points = cloud[mask > 0]
        if marker_points.shape[0] < 3: return None
        tvec = np.nanmean(marker_points, axis=0)


        mask[:] = 0
        cv2.line(mask, tuple(marker.center - marker.axisX), tuple(marker.center + marker.axisX), 255, AXIS_STRIPE)
        axisX_points = cloud[mask > 0]
        axisX_points = axisX_points[np.isfinite(axisX_points).all(axis=1)]
        if axisX_points.shape[0] < 3: return None

        M = np.sum([np.outer(p,p) for p in axisX_points - tvec], axis=0)
        _,_,v = linalg.svd(M)
        axisX = v[0] if np.sign(marker.axisX[0]) == np.sign(v[0,0]) else -v[0]


        mask[:] = 0
        cv2.line(mask, tuple(marker.center - marker.axisY), tuple(marker.center + marker.axisY), 255, AXIS_STRIPE)
        axisY_points = cloud[mask > 0]
        axisY_points = axisY_points[np.isfinite(axisY_points).all(axis=1)]
        if axisY_points.shape[0] < 3: return None

        M = np.sum([np.outer(p,p) for p in axisY_points - tvec], axis=0)
        _,_,v = linalg.svd(M)
        axisY = -v[0] if np.sign(marker.axisY[1]) == np.sign(v[0,1]) else v[0]


        axisZ = np.cross(axisX, axisY)

        R = np.vstack( (axisX, axisY, axisZ) ).T
        rvec,_ = cv2.Rodrigues(R)

        return rvec, tvec[:,np.newaxis]



