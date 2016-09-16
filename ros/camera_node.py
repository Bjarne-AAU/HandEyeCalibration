#!/usr/bin/env python

import rospy
import cv2
import numpy as np

import conversions as convert

from sensor_msgs.msg import Image as ImageMsg

from nodes.camera import Camera
from nodes.display import ImageDisplay

class CameraNode(object):

    def __init__(self, device=-1):
        rospy.init_node("camera_node", anonymous=True)
        self._pubImage = rospy.Publisher("image", ImageMsg, queue_size=1)
        self._timer = None
        self._camera = Camera(device)
        self._display = ImageDisplay(open=False)

    def run(self):
        self._display.open()
        self._timer = rospy.Timer(rospy.Duration(0.05), self.__callback)
        rospy.spin()

    def __callback(self, event):
        if not self._display.isOpen():
            self._camera.close()
            self._timer.shutdown()
            rospy.signal_shutdown("The End")
            return

        frame_id = "kittingbox_camera_rgb_optical_frame"
        stamp = rospy.Time.now()

        image = self._camera.getFrame()
        self._display.show(image)
        imageMsg = convert.matrix2image(image, frame_id, stamp)

        self._pubImage.publish(imageMsg)




if __name__ == '__main__':
    node = CameraNode(-1)
    node.run()
