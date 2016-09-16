#!/usr/bin/env python

import rospy
import cv2
import numpy as np

import tf
import tf_conversions

from cv_bridge import CvBridge

from geometry_msgs.msg import PoseStamped as PoseStampedMsg
from sensor_msgs.msg import Image as ImageMsg

from nodes.display import ImageDisplay

from detector.chessboard_detector import DetectorChessboard

from EyeOnHandCalibration import EyeOnHandCalibration
from EyeOnHandCalibration import TYPE
from EyeOnHandCalibration import METHOD



class ChessboardCalibratorNode(object):

    def __init__(self):
        rospy.init_node("chessboard_calibrator_node", anonymous=True)

        self._transformer = tf.TransformListener()
        # rospy.sleep(1)
        self._bridge = CvBridge()
        self._detector = DetectorChessboard()
        self._display = ImageDisplay(open=False)

        self._pubPalm   = rospy.Publisher("pose_palm",   PoseStampedMsg, queue_size=1)
        self._pubMarker = rospy.Publisher("pose_marker", PoseStampedMsg, queue_size=1)
        self._pubCamera = rospy.Publisher("pose_camera", PoseStampedMsg, queue_size=1)

        self._subImage  = rospy.Subscriber("kittingbox_camera/rgb/image_raw", ImageMsg, self.__callback)

    def run(self):
        self._display.open()
        rospy.spin()

    def marker2pose(self, marker, frame_id, time):
        msg = PoseStampedMsg()
        T = np.identity(4)
        (R,_) = cv2.Rodrigues(marker.Rvec)
        T[0:3,0:3] = R
        T[0:3,3:4] = marker.Tvec
        T = tf_conversions.fromMatrix(T)
        msg.pose = tf_conversions.toMsg(T)
        msg.header.frame_id = frame_id
        msg.header.stamp = time
        return msg

    def transform2pose(self, transform, frame_id, time):
        msg = PoseStampedMsg()
        T = tf_conversions.fromTf(transform)
        msg.pose = tf_conversions.toMsg(T)
        msg.header.frame_id = frame_id
        msg.header.stamp = time
        return msg

    def matrix2pose(self, matrix, frame_id, time):
        msg = PoseStampedMsg()
        T = tf_conversions.fromMatrix(matrix)
        msg.pose = tf_conversions.toMsg(T)
        msg.header.frame_id = frame_id
        msg.header.stamp = time
        return msg

    def pose2matrix(self, pose):
        T = tf_conversions.fromMsg(pose.pose)
        T = tf_conversions.toMatrix(T)
        return T


    def __callback(self, data):
        frame_id = data.header.frame_id
        stamp = data.header.stamp

        image = self._bridge.imgmsg_to_cv2(data, "bgr8")
        corners = self._detector.process(image)
        image = self._detector.drawChessboard(image, corners)

        self._display.show(image)
        
        if corners is None: return
        
        T = self._detector.solve(corners)
        markerPose = self.matrix2pose(T, frame_id, stamp)
        self._pubMarker.publish(markerPose)        
            
        



if __name__ == '__main__':
    node = ChessboardCalibratorNode()
    node.run()
