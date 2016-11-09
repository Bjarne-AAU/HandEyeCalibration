# /usr/bin/env python

import numpy as np
import scipy.linalg as linalg
import tools.transform as tf


class MetaClassDetector(type):
    def __repr__(self):
        return self.__name__

class Calibrator(object):
    __metaclass__ = MetaClassDetector
    def __repr__(self):
        return self.name

    def __init__(self, solver):
        self._solver = solver
        self._lastSample = (None, None)

    @property
    def name(self):
        return type(self).__name__

    def setReferenceFrame(self, frame):
        self._frame_reference = frame

    def setSolutionFrame(self, frame):
        self._frame_solution = frame

    def getReferenceFrame(self):
        return self._frame_reference

    def getSolutionFrame(self):
        return self._frame_solution

    def setSolver(self, solver):
        self._solver = solver

    def isSimilarSample(self, A, B):
        if A is None or B is None: return False
        angle, distance = tf.smallest_error(A, B)
        return distance < 0.05 and angle < 5.0

    def addSample(self, A, B):
        oldA, oldB = self._lastSample
        if self.isSimilarSample(A, oldA):
            return False

        newA, newB = self._prepareSample(oldA, A, oldB, B)
        if newA is not None and newB is not None:
            self._solver.addSample(newA, newB)
            print("Added sample " + str(self._solver.N))

        self._lastSample = (A, B)
        return True

    def _prepareSample(self, oldA, A, oldB, B):
        raise NotImplementedError()

    def reset(self):
        self._solver.reset()

    def solve(self):
        if self._solver.N < 5: return None
        _, Y = self._solver.solve()
        return Y
