# /usr/bin/env python

import numpy as np
import scipy.linalg as linalg

from tools.transform import *

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

        X = vector_matrix(v[0,0:9])
        X = orthonormalize_rotation(X)

        Y = vector_matrix(u[0:9,0])
        Y = orthonormalize_rotation(Y)

        return (X, Y)
