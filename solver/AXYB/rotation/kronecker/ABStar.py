# /usr/bin/env python

import numpy as np
import scipy.linalg as linalg

from tools.tools import *

from solver.AXYB.SolverAXYB import SolverAXYB

class ABStar(SolverAXYB):

    @property
    def shortName(self):
        return "YKronR*"

    def solveRotation(self):
        N = self.N
        I = np.identity(9)
        M = np.zeros( (9,9) )

        for i in range(N):
            RA = self.RA(i)
            RB = self.RB(i)

            M += linalg.kron(RA, RB)

        [u,s,v] = linalg.svd(M)

        RX = np.reshape(v[0,:], (3,3))
        RY = np.reshape(u[:,0], (3,3))

        X = np.identity(4)
        X[0:3, 0:3] = orthonormalize_rotation(RX)

        Y = np.identity(4)
        Y[0:3, 0:3] = orthonormalize_rotation(RY)

        return (X, Y)
