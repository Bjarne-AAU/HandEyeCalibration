import cv2

class ImageDisplay(object):

    def __init__(self, name = "image", open=True):
        self._isOpen = False
        self._windowName = name
        # cv2.startWindowThread()
        if open: self.open()

    def __isWindowOpen(self):
        return cv2.getWindowProperty(self._windowName, cv2.WND_PROP_ASPECT_RATIO) > 0

    def isOpen(self):
        return self._isOpen

    def open(self):
        cv2.namedWindow(self._windowName)
        cv2.moveWindow(self._windowName, 0, 0)
        self._isOpen = True

    def close(self):
        cv2.destroyWindow(self._windowName)
        self._isOpen = False

    def show(self, image):
        if not self.isOpen(): return False

        cv2.imshow(self._windowName, image)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or not self.__isWindowOpen():
            self.close()
            return False
        return True

    def spin(self, wait=10):
        if not self.isOpen(): return False
        key = cv2.waitKey(wait) & 0xFF
        if key == 27 or not self.__isWindowOpen():
            self.close()
            return False
        return True
