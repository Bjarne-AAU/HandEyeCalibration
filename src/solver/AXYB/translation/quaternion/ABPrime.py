# /usr/bin/env python

import numpy as np
import scipy.linalg as linalg

from tools.transform import *

from solver.AXYB.SolverAXYB import SolverAXYB

class ABPrime(SolverAXYB):

    @property
    def shortName(self):
        return "YQuatT'"

    def solve(self):
        N = self.N
        I = np.identity(4)
        M = np.zeros( (12,12) )
        scale = 1.0

        for i in range(N):
            qA = quaternion_from_matrix(self.RA(i))
            qB = quaternion_from_matrix(self.RB(i))
            qtA = quaternion_from_imaginary(self.tA(i))
            qtB = quaternion_from_imaginary(self.tB(i))

            RA = matrix_representation(qA, PLUS)
            RB = matrix_representation(qB, MINUS)
            tA = matrix_representation(qtA, PLUS) * scale
            tB = matrix_representation(qtB, MINUS) * scale

            C = np.zeros( (4,12) )
            C[0:4,  0: 4] = tB - tA
            C[0:4,  4: 8] = I
            C[0:4,  8:12] = -RB.T.dot(RA)

            M += C.T.dot(C)

        [u,s,v] = linalg.svd(M)

        q,t = computeDualQuaternionNullSpace(v[-1,0:8], v[-2,0:8])

        E = quaternion_matrix_representation(q, MINUS).T
        t = E.dot(t) / scale

        Y = quaternion_matrix(q)
        Y[0:3,3] = t[1:4]

        X = np.identity(4)

        return (X,Y)

