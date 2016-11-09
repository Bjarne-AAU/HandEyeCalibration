# /usr/bin/env python

import numpy as np
import scipy.linalg as linalg

from tools.transform import *

from solver.AXXB.SolverAXXB import SolverAXXB

class ABStar(SolverAXXB):

    @property
    def shortName(self):
        return "XKronR*"

    def solveRotation(self):
        N = self.N
        I = np.identity(9)
        M = np.zeros( (9,9) )

        for i in range(N):
            RA = self.RA(i)
            RB = self.RB(i)

            M += linalg.kron(RA, RB)

        [u,s,v] = linalg.svd(M)

        X = vector_matrix(v[0,:])
        X = orthonormalize_rotation(X)

        return (X, X)
