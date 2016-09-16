# /usr/bin/env python

import numpy as np
import scipy.linalg as linalg

from solver.SolverInterface import SolverInterface


class SolverAXYB(SolverInterface):

    def __init__(self):
        super(SolverAXYB, self).__init__()

    @property
    def shortName(self):
        return "AXYB"

    def solve(self):
        # super(SolverAXYB, self).solve()
        (RX, RY) = self.solveRotation()
        (TX, TY) = self.solveTranslation(RY)
        TX[:3,:3] = RX[:3,:3]
        TY[:3,:3] = RY[:3,:3]
        return (TX, TY)

    def solveRotation(self):
        return (np.identity(4), np.identity(4))

    def solveTranslation(self, R):
        N = self.N
        R = R[0:3, 0:3]
        I = np.identity(3)

        A = np.zeros( (N,3,6) )
        b = np.zeros( (N,3,1) )

        for i in range(N):
            RA = self._AA[i, 0:3, 0:3]
            tA = self._AA[i, 0:3, 3:4].T
            tB = self._BB[i, 0:3, 3:4].T
            A[i, :, :] = np.hstack( (-RA, I) )
            b[i, :, 0:1] = tA.T - R.dot(tB.T)

        A = np.reshape(A, (N*3, -1))
        b = np.reshape(b, (N*3, -1))
        x = linalg.lstsq(A, b)[0]

        TX = np.identity(4)
        TX[0:3, 3] = x[0:3].T

        TY = np.identity(4)
        TY[0:3, 3] = x[3:6].T
        return (TX, TY)


    def solveX(self):
        (X,Y) = self.solve()
        return X

    def solveY(self):
        (X,Y) = self.solve()
        return Y

    def solveRotationX(self):
        (RX, RY) = self.solveRotation()
        return RX

    def solveTranslationX(self, R):
        (TX, TY) = self.solveTranslation(R)
        return TX

    def solveRotationY(self):
        (RX, RY) = self.solveRotation()
        return RY

    def solveTranslationY(self, R):
        (TX, TY) = self.solveTranslation(R)
        return TY


if __name__ == "__main__":
    pass
