# /usr/bin/env python

import cv2

class Camera(object):

    def __init__(self, device = -1):
        self._size = [0,0]
        self._cam = None
        self._device = device
        self._callback = None
        self.open(device)

    def open(self, device = -1):
        if self.isOpen() and self._device == device:
            return

        self.close()
        self._device = device
        self._cam = cv2.VideoCapture(device)
        if not self.isOpen():
            raise Exception("Could not open camera device: " + str(device))
        [success, frame] = self._cam.read()
        if not success:
            raise Exception("Could not read from camera device: " + str(device))

        self._size = frame.shape[0:2]


    def close(self):
        if self.isOpen():
            self._cam.release()


    def isOpen(self):
        return self._cam and self._cam.isOpened()


    def setCallback(self, callback):
        self._callback = callback

    def getImageSize(self):
        return self._size

    def getFrame(self):
        if not self.isOpen(): return None
        [success, frame] = self._cam.read()
        return frame


    def processFrame(self):
        frame = self.getFrame()
        if frame is None:
            return False

        if self._callback is not None:
            frame = self._callback(frame)

        return frame is not None and frame is not False


    def run(self):
        while self.processFrame():
            pass
