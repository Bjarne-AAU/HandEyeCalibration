#!/usr/bin/env python

import cv2
import numpy as np
import scipy.linalg as linalg

import tf_conversions

from cv_bridge import CvBridge

from geometry_msgs.msg import PoseStamped as PoseStampedMsg
from sensor_msgs.msg import Image as ImageMsg



__bridge = CvBridge()

def axis2matrix(rvec, tvec):
    T = np.identity(4)
    (R,_) = cv2.Rodrigues(rvec)
    T[0:3,0:3] = R
    T[0:3,3:4] = tvec
    return T

def matrix2axis(matrix):
    # logR = linalg.logm(matrix[0:3, 0:3]).real
    # rvec = np.array([logR[2,1], logR[0,2], logR[1,0]])
    # rvec = rvec[:, np.newaxis].copy()
    (rvec,_) = cv2.Rodrigues(matrix[0:3,0:3])
    tvec = matrix[0:3,3:4]
    return (rvec, tvec)


def axis2pose(rvec, tvec, frame_id, time):
    msg = PoseStampedMsg()
    T = axis2matrix(rvec, tvec)
    T = tf_conversions.fromMatrix(T)
    msg.pose = tf_conversions.toMsg(T)
    msg.header.frame_id = frame_id
    msg.header.stamp = time
    return msg

def transform2pose(transform, frame_id, time):
    msg = PoseStampedMsg()
    T = tf_conversions.fromTf(transform)
    msg.pose = tf_conversions.toMsg(T)
    msg.header.frame_id = frame_id
    msg.header.stamp = time
    return msg

def matrix2pose(matrix, frame_id, time):
    msg = PoseStampedMsg()
    T = tf_conversions.fromMatrix(matrix)
    msg.pose = tf_conversions.toMsg(T)
    msg.header.frame_id = frame_id
    msg.header.stamp = time
    return msg

def pose2matrix(pose):
    T = tf_conversions.fromMsg(pose.pose)
    T = tf_conversions.toMatrix(T)
    return T


def matrix2image(matrix, frame_id, time, encoding = "bgr8"):
    global __bridge
    msg = __bridge.cv2_to_imgmsg(matrix, encoding)
    msg.header.frame_id = frame_id
    msg.header.stamp = time
    return msg

def image2matrix(image, encoding = "bgr8"):
    global __bridge
    return __bridge.imgmsg_to_cv2(image, encoding)
