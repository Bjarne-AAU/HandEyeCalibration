# /usr/bin/env python

import numpy as np
import scipy.linalg as linalg

from tools.tools import *

from solver.AXYB.SolverAXYB import SolverAXYB

class AB(SolverAXYB):

    @property
    def shortName(self):
        return "YKronR"

    def solveRotation(self):
        N = self.N
        I = np.identity(9)
        M = np.zeros( (18,18) )

        for i in range(N):
            RA = self.RA(i)
            RB = self.RB(i)

            C = np.zeros( (9,18) )
            # C[0:9,   0:9 ] = linalg.kron(RA, I)
            # C[0:9,   9:18] = -linalg.kron(-I, RB.T)
            C[0:9,   0:9 ] = linalg.kron(RA, RB)
            C[0:9,   9:18] = -I

            M += C.T.dot(C)

        [u,s,v] = linalg.svd(M)

        RX = np.reshape(v[-1,0:9 ], (3,3))
        RY = np.reshape(v[-1,9:18], (3,3))

        X = np.identity(4)
        X[0:3, 0:3] = orthonormalize_rotation(RX)

        Y = np.identity(4)
        Y[0:3, 0:3] = orthonormalize_rotation(RY)

        return (X, Y)
