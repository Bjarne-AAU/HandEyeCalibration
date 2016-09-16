# /usr/bin/env python

import numpy as np
import scipy.linalg as linalg

from tools.tools import *

from solver.AXXB.SolverAXXB import SolverAXXB

class XX(SolverAXXB):

    @property
    def shortName(self):
        return "XKronRX"

    def splitMatrix(self, M, nrows, ncols):
        h = M.shape[0]
        return (M.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))

    def solveRotation(self):
        N = self.N
        I = np.identity(9)
        M = np.zeros( (9,9) )

        for i in range(N):
            RA = self.RA(i)
            RB = self.RB(i)

            M += np.outer(RA.T.ravel(), RB.T.ravel())

        [u,s,v] = linalg.svd(M)

        # R = self.splitMatrix(u.dot(v), 3, 3).sum(axis=0)
        # R = self.splitMatrix(u.dot(v), 3, 3).sum(axis=1).sum(axis=1).reshape((3,3))
        R = u.dot(v)[3:6,3:6]

        X = np.identity(4)
        X[0:3,0:3] = orthonormalize_rotation(R)

        return (X, X)
