# /usr/bin/env python

import numpy as np
import scipy.linalg as linalg

from tools.tools import *

from solver.AXXB.SolverAXXB import SolverAXXB


class XX(SolverAXXB):

    @property
    def shortName(self):
        return "XAxisRX"

    def solveRotation(self):
        N = self.N
        M = np.zeros( (3,3) )

        for i in range(N):
            AA = axis_angle_from_matrix(self.RA(i))[0]
            AB = axis_angle_from_matrix(self.RB(i))[0]

            M += AA.dot(AB.T)

        # Park's implementation
        # [u,s,v] = linalg.svd(M.dot(M.T))
        # s = np.diag(s**-0.5)
        # R = v.T.dot(s).dot(v).dot(M)

        [u,s,v] = linalg.svd(M)
        R = u.dot(v)

        X = np.identity(4)
        X[0:3,0:3] = R

        return (X, X)
