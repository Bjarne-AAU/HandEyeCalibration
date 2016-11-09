# /usr/bin/env python

import numpy as np

class MetaClassSolver(type):
    def __repr__(self):
        return self.__name__

class SolverInterface(object):
    __metaclass__ = MetaClassSolver
    def __repr__(self):
        return self.shortName

    A = 0
    B = 1

    ROTATION = 0
    TRANSLATION = 1

    def __init__(self):
        self._AA = np.zeros( (0,4,4) )
        self._BB = np.zeros( (0,4,4) )

    @property
    def shortName(self):
        return "Interface"

    @property
    def name(self):
        modules = type(self).__module__.split(".")
        n = modules[1] + " "
        for m in modules[2:-1]:
            n += "|" + m.capitalize().ljust(16, ' ')
        n += "|" + type(self).__name__
        return n

    def log(self, msg):
        print("[" + self.name + "] " + msg)

    @property
    def N(self):
        return np.shape(self._AA)[0]

    def sample(self, i, which, type=None):
        data = self._BB[i] if which==self.B else self._AA[i]
        if type == self.ROTATION: return data[:3,:3]
        elif type == self.TRANSLATION: return data[:3,3:4]
        else: return data

    def RA(self, i): return self.sample(i, self.A, self.ROTATION)
    def tA(self, i): return self.sample(i, self.A, self.TRANSLATION)
    def RB(self, i): return self.sample(i, self.B, self.ROTATION)
    def tB(self, i): return self.sample(i, self.B, self.TRANSLATION)


    def addSample(self, A, B):
        self._AA = np.append( self._AA, [A], axis=0 )
        self._BB = np.append( self._BB, [B], axis=0 )

    def reset(self):
        self._AA = np.zeros( (0,4,4) )
        self._BB = np.zeros( (0,4,4) )


    def solve(self):
        # self.log("Solving with " + str(self.N)+ " samples")
        R = self.solveRotation()
        (RX, RY) = R if isinstance(R, tuple) else (R, R)
        t = self.solveTranslation(RY)
        (tX, tY) = t if isinstance(t, tuple) else (t, t)
        tX[:3,:3] = RX[:3,:3]
        tY[:3,:3] = RY[:3,:3]
        return (tX, tY)

    def solveRotation(self):
        I = np.identity(4)
        return (I, I)

    def solveTranslation(self, R):
        return (R, R)
