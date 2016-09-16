#!/usr/bin/env python

import math
import struct
import rospy
import cv2
import numpy as np
import aruco

import tf
import ros.conversions as convert

from geometry_msgs.msg import PoseStamped as PoseMsg
from sensor_msgs.msg import Image as ImageMsg

from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2 as PointCloud2Msg

from nodes.display import ImageDisplay

from detector.chessboard_detector import ChessboardModel
from estimator.chessboard_pose_estimator import ChessboardPoseEstimator
from estimator.aruco_pose_estimator import ArucoPoseEstimator

from EyeOnHandCalibration import EyeOnHandCalibration
from EyeOnHandCalibration import TYPE
from EyeOnHandCalibration import METHOD

from EyeInHandCalibration import EyeInHandCalibration
from EyeInHandCalibration import TYPE
from EyeInHandCalibration import METHOD


class CalibratorNode(object):

    def __init__(self, detector, solver):
        rospy.init_node("calibrator_node", anonymous=True)

        self._transformer = tf.TransformListener()
        # rospy.sleep(1)
        self._estimator = detector
        self._calibrator = solver
        self._display = ImageDisplay()

        self._pubPalm   = rospy.Publisher("pose_palm",   PoseMsg, queue_size=1)
        self._pubMarker = rospy.Publisher("pose_marker", PoseMsg, queue_size=1)
        self._pubCamera = rospy.Publisher("pose_camera", PoseMsg, queue_size=1)

        # self._subImage  = rospy.Subscriber("/kittingbox_camera/rgb/image_raw", ImageMsg, self.__callback)

        self._subPCL  = rospy.Subscriber("/kittingbox_camera/depth_registered/points", PointCloud2Msg, self.__callbackPCL)
        self._pubPCL  = rospy.Publisher("points", PointCloud2Msg, queue_size=1)

    def run(self):
        self._display.open()
        rospy.spin()


    def to_rgb(self, rgb_floats):
        rgbs = np.array(struct.unpack('B'*4*len(rgb_floats), struct.pack('f'*len(rgb_floats), *rgb_floats)))
        r = rgbs[0::4]
        g = rgbs[1::4]
        b = rgbs[2::4]
        rgbs = zip(r,g,b)

        return np.array(rgbs, dtype=np.uint8)

    def __callbackPCL(self, data):
        if not self._display.isOpen():
            rospy.signal_shutdown("The End")
            return

        frame_id = data.header.frame_id
        stamp = data.header.stamp


        cloud = np.array(list(pc2.read_points(data)))
        rgbs = self.to_rgb(cloud[:,3])

        cloud = cloud.reshape(data.height, data.width, -1)
        image = rgbs.reshape(data.height, data.width, -1)

        marker = self._estimator.process(image)

        image = self._estimator.draw(image, marker)
        self._display.show(image)

        if marker is None: return

        center = np.round(marker.getCenter())
        size = 5
        roi = cloud[center[1]-size:center[1]+size, center[0]-size:center[0]+size, 0:3]
        t = np.nanmean(roi.reshape(-1,3), axis=0)
        if np.any(np.isnan(t)): return
        t = np.copy(t[:, np.newaxis])
        marker.Tvec = t

        markerPose = convert.axis2pose(marker.Rvec, marker.Tvec, frame_id, stamp)
        self._pubMarker.publish(markerPose)

        try:
            self._transformer.waitForTransform("base_link", "palm", stamp, rospy.Duration(0.5))
            transform = self._transformer.lookupTransform("base_link", "palm", stamp)
        except Exception, e:
            print(e)
            return

        palmPose = convert.transform2pose(transform, "base_link", stamp)
        self._pubPalm.publish(palmPose)


        A = convert.pose2matrix(palmPose)
        B = convert.pose2matrix(markerPose)
        self._calibrator.addSample(A, B)

        if self._calibrator.N > 3:
            T = self._calibrator.solve()
            cameraPose = convert.matrix2pose(T, "base_link", stamp)
            self._pubCamera.publish(cameraPose)


    def __callback(self, data):
        if not self._display.isOpen():
            rospy.signal_shutdown("The End")
            return

        frame_id = data.header.frame_id
        stamp = data.header.stamp

        image = convert.image2matrix(data)
        marker = self._estimator.process(image)

        image = self._estimator.draw(image, marker)
        self._display.show(image)

        if marker is None: return

        markerPose = convert.axis2pose(marker.Rvec, marker.Tvec, frame_id, stamp)
        self._pubMarker.publish(markerPose)

        try:
            self._transformer.waitForTransform("base_link", "palm", stamp, rospy.Duration(0.5))
            transform = self._transformer.lookupTransform("base_link", "palm", stamp)
        except Exception, e:
            print(e)
            return

        palmPose = convert.transform2pose(transform, "base_link", stamp)
        self._pubPalm.publish(palmPose)


        A = convert.pose2matrix(palmPose)
        B = convert.pose2matrix(markerPose)
        self._calibrator.addSample(A, B)

        if self._calibrator.N > 3:
            T = self._calibrator.solve()
            cameraPose = convert.matrix2pose(T, "base_link", stamp)
            self._pubCamera.publish(cameraPose)


if __name__ == '__main__':
    detector = ArucoPoseEstimator(marker_size = 0.04)
    # detector = ChessboardPoseEstimator(ChessboardModel(7, 5, 0.03))
    solver = EyeOnHandCalibration(name="Grossmann1Step")#, type=TYPE.AXYB, method=METHOD.QUAT)

    # # detector = ArucoPoseEstimator(marker_size = 0.04)
    # detector = ChessboardPoseEstimator(ChessboardModel(7, 5, 0.03))
    # solver = EyeInHandCalibration(name="Shah")#, type=TYPE.AXYB, method=METHOD.QUAT)

    node = CalibratorNode(detector, solver)
    node.run()
