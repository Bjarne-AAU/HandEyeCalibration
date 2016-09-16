# /usr/bin/env python

import numpy as np
import scipy.linalg as linalg

from tools.tools import *

from solver.AXXB.SolverAXXB import SolverAXXB


class XX(SolverAXXB):

    @property
    def shortName(self):
        return "XQuatRX"

    def solveRotation(self):
        N = self.N
        M = np.zeros( (4,4) )

        for i in range(N):
            QA = quaternion_from_matrix(self.RA(i))
            QB = quaternion_from_matrix(self.RB(i))

            M += QA.dot(QB.T)

        [u,s,v] = linalg.svd(M)
        R = u.dot(v)

        X = np.identity(4)
        X[0:3,0:3] = R[1:4,1:4]

        return (X, X)

