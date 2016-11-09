import numpy as np
import scipy.linalg as linalg

class Marker(object):

    def __init__(self, points):
        self._points = np.array(points)

    @property
    def points(self):
        return self._points

    @property
    def box(self):
        return self._points

    @property
    def shape(self):
        return self._points

    @property
    def center(self):
        return np.mean(self._points, axis=0)

    @property
    def axisX(self):
        tl,tr,br,bl = self.box
        axis = (tr+br)/2.0 - (tl+bl)/2.0
        return axis/2.0

    @property
    def axisY(self):
        tl,tr,br,bl = self.box
        axis = (bl+br)/2.0 - (tl+tr)/2.0
        return axis/2.0


class MetaClassDetector(type):
    def __repr__(self):
        return self.__name__

class Detector(object):
    __metaclass__ = MetaClassDetector
    def __repr__(self):
        return self.name

    def __init__(self, model):
        self._model = model

    @property
    def name(self):
        return type(self).__name__

    @property
    def model(self):
        return self._model

    def process(self, frame, draw=True):
        marker = self._detect(frame)
        if draw and marker is not None:
            frame = self._draw(frame, marker)
        return marker

    def _draw(self, frame):
        return frame

    def _detect(self, frame):
        raise NotImplementedError()
