# /usr/bin/env python

import numpy as np
import scipy.linalg as linalg

from tools.transform import *

from solver.AXXB.SolverAXXB import SolverAXXB

class AB(SolverAXXB):

    @property
    def shortName(self):
        return "XKronRT"

    def solveRotation(self):
        N = self.N
        I = np.identity(3)
        II = np.identity(9)
        A = np.zeros( (N,12,12) )
        b = np.zeros( (N,12,1) )

        for i in range(N):
            RA = self.RA(i)
            RB = self.RB(i)
            tA = self.tA(i)
            tB = self.tB(i)

            A[i, 0:9,  0:9 ] = II - linalg.kron(RA, RB)
            A[i, 9:12, 0:9 ] = linalg.kron(I, tB.T)
            A[i, 9:12, 9:12] = I - RA

            b[i, 9:12, 0:1] = tA

            b[i, 0:9 ] = 0
            b[i, 9:12] = tA

        A = np.reshape(A, (A.shape[0]*A.shape[1],-1))
        b = np.reshape(b, (b.shape[0]*b.shape[1],-1))
        x = linalg.lstsq(A, b)[0]

        X = vector_matrix(x[0:9])
        X[0:3, 3:4] = x[9:12]
        X = orthonormalize_rotation(X)

        return (X, X)
