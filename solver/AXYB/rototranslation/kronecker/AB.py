# /usr/bin/env python

import numpy as np
import scipy.linalg as linalg

from tools.tools import *

from solver.AXYB.SolverAXYB import SolverAXYB

class AB(SolverAXYB):

    @property
    def shortName(self):
        return "YKronRT"

    def solveRotation(self):
        N = self.N
        I = np.identity(3)
        II = np.identity(9)
        A = np.zeros( (N,12,24) )
        b = np.zeros( (N,12,1) )

        for i in range(N):
            RA = self.RA(i)
            RB = self.RB(i)
            tA = self.tA(i)
            tB = self.tB(i)

            # A[i, 0:9,   0:9 ] = linalg.kron(RA, I)
            # A[i, 0:9,   9:18] = linalg.kron(-I, RB.T)
            A[i, 0:9,   0:9 ] = linalg.kron(RA, RB)
            A[i, 0:9,   9:18] = -II
            A[i, 9:12,  9:18] = linalg.kron(I, tB.T)
            A[i, 9:12, 18:21] = -RA
            A[i, 9:12, 21:24] = I

            b[i, 9:12] = tA

        A = np.reshape(A, (N*12,-1))
        b = np.reshape(b, (N*12,-1))
        x = linalg.lstsq(A, b)[0]

        RX = np.reshape(x[0:9], (3,3))
        RY = np.reshape(x[9:18], (3,3))

        X = np.identity(4)
        X[0:3, 3:4] = x[18:21]
        X[0:3, 0:3] = orthonormalize_rotation(RX)

        Y = np.identity(4)
        Y[0:3, 3:4] = x[21:24]
        Y[0:3, 0:3] = orthonormalize_rotation(RY)

        return (X, Y)
