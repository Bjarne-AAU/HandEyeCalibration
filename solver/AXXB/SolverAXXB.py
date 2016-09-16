# /usr/bin/env python

import numpy as np
import scipy.linalg as linalg

from solver.SolverInterface import SolverInterface


class SolverAXXB(SolverInterface):

    def __init__(self):
        super(SolverAXXB, self).__init__()

    @property
    def shortName(self):
        return "AXYB"

    def solveTranslation(self, R):
        N = self.N
        R = R[0:3, 0:3]
        I = np.identity(3)

        A = np.zeros( (N,3,3) )
        b = np.zeros( (N,3,1) )

        for i in range(N):
            RA = self.RA(i)
            tA = self.tA(i)
            tB = self.tB(i)
            A[i, :, :]   = I - RA
            b[i, :, 0:1] = tA - R.dot(tB)

        A = np.reshape(A, (A.shape[0]*A.shape[1], -1))
        b = np.reshape(b, (b.shape[0]*b.shape[1], -1))
        x = linalg.lstsq(A, b)[0]

        TX = np.identity(4)
        TX[0:3, 3:4] = x

        return (TX, TX)
