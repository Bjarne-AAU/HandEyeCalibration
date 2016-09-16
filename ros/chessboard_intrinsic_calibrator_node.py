#!/usr/bin/env python

import rospy
import cv2
import numpy as np

import tf
import conversions as convert


from geometry_msgs.msg import PoseStamped as PoseStampedMsg
from sensor_msgs.msg import Image as ImageMsg

from nodes.display import ImageDisplay

from detector.chessboard_detector import ChessboardModel
from detector.chessboard_detector import ChessboardDetector



class ChessboardIntrinsicCalibratorNode(object):

    def __init__(self):
        rospy.init_node("chessboard_intrinsic_calibrator_node", anonymous=True)

        self._model = ChessboardModel(5, 5, 0.045)
        self._detector = ChessboardDetector(self._model)
        self._display = ImageDisplay(open=False)

        self._counter = 0


        self._samplesObj = []
        self._samplesImg = []

        self._pubPalm   = rospy.Publisher("pose_palm",   PoseStampedMsg, queue_size=1)
        self._pubMarker = rospy.Publisher("pose_marker", PoseStampedMsg, queue_size=1)
        self._pubCamera = rospy.Publisher("pose_camera", PoseStampedMsg, queue_size=1)

        self._subImage  = rospy.Subscriber("image", ImageMsg, self.__callback)

    def run(self):
        self._display.open()
        rospy.spin()

    def __callback(self, data):

        self._counter += 1
        if self._counter % 100 != 0: return

        frame_id = data.header.frame_id
        stamp = data.header.stamp

        frame_id = data.header.frame_id
        stamp = data.header.stamp

        image = convert.image2matrix(data)
        marker = self._detector.process(image)

        image = self._detector.draw(image, marker)
        self._display.show(image)

        if marker is None: return
        if len(marker) != self._model.tiles: return

        self._samplesObj.append(self._model.points)
        self._samplesImg.append(marker)

        w = image.shape[1]
        h = image.shape[0]

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self._samplesObj, self._samplesImg, (w,h), None, None, flags=cv2.CALIB_FIX_PRINCIPAL_POINT)

        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        image = cv2.undistort(image, mtx, dist, None, newcameramtx)
        #x,y,w,h = roi
        #image = image[y:y+h, x:x+w]

        self._display.show(image)

        print(mtx)
        print(dist)

        tvecs = np.array(tvecs)
        rvecs = np.array(rvecs)
        tvec = np.mean(tvecs, axis=0)
        rvec = rvecs[-1]

        T = np.identity(4)
        (R,_) = cv2.Rodrigues(rvec)
        T[0:3,0:3] = R
        T[0:3,3:4] = tvec

        markerPose = convert.matrix2pose(T, frame_id, stamp)
#        print(markerPose)
        self._pubMarker.publish(markerPose)





if __name__ == '__main__':
    node = ChessboardIntrinsicCalibratorNode()
    node.run()
