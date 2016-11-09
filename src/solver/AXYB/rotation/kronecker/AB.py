# /usr/bin/env python

import numpy as np
import scipy.linalg as linalg

from tools.transform import *

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

        X = vector_matrix(v[-1,0:9])
        X = orthonormalize_rotation(X)

        Y = vector_matrix(v[-1,9:18])
        Y = orthonormalize_rotation(Y)

        return (X, Y)
