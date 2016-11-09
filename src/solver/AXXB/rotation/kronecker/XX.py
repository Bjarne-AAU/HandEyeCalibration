# /usr/bin/env python

import numpy as np
import scipy.linalg as linalg

from tools.transform import *

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

            vA = vector_from_matrix(RA)
            vB = vector_from_matrix(RB)

            M += vA.dot(vB.T)

        [u,s,v] = linalg.svd(M)

        # R = self.splitMatrix(u.dot(v), 3, 3).sum(axis=0)
        # R = self.splitMatrix(u.dot(v), 3, 3).sum(axis=1).sum(axis=1).reshape((3,3))
        R = u.dot(v)[3:6,3:6]
        X = transformation_from_rotation(R)
        X = orthonormalize_rotation(X)

        return (X, X)
