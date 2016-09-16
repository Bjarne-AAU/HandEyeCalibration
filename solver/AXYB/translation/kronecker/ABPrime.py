# /usr/bin/env python

import numpy as np
import scipy.linalg as linalg

from tools.tools import *

from solver.AXYB.SolverAXYB import SolverAXYB

class ABPrime(SolverAXYB):

    @property
    def shortName(self):
        return "YKronT'"

    def solveRotation(self):
        N = self.N
        I = np.identity(3)
        A = np.zeros( (N,3,15) )
        b = np.zeros( (N,3,1) )

        for i in range(N):
            RA = self.RA(i)
            # RB = self.RB(i)
            tA = self.tA(i)
            tB = self.tB(i)

            A[i, 0:3,  0:3] = -RA
            A[i, 0:3,  3:6] = I
            A[i, 0:3, 6:15] = linalg.kron(I, tB.T)

            b[i] = tA

        A = np.reshape(A, (N*3, -1))
        b = np.reshape(b, (N*3, -1))
        x = linalg.lstsq(A, b)[0]

        X = np.identity(4)
        X[0:3,3:4] = x[0:3]

        Y = np.identity(4)
        Y[0:3,3:4] = x[3:6]
        Y[0:3,0:3] = orthonormalize_rotation(np.reshape(x[6:15], (3,3)))

        (TX, TY) = self.solveTranslation(Y)
        return (TX.dot(X), TY.dot(Y))
