# /usr/bin/env python

import numpy as np
import scipy.linalg as linalg

from tools.transform import *

from solver.AXXB.SolverAXXB import SolverAXXB

class ABStar(SolverAXXB):

    @property
    def shortName(self):
        return "XQuatRT*"

    def solve(self):
        N = self.N
        I = np.identity(4)
        M = np.zeros( (8,8) )
        scale = 0.5

        for i in range(N):
            qA = quaternion_from_matrix(self.RA(i))
            qB = quaternion_from_matrix(self.RB(i))
            qtA = quaternion_from_imaginary(self.tA(i))
            qtB = quaternion_from_imaginary(self.tB(i))

            RA = matrix_representation(qA, PLUS)
            RB = matrix_representation(qB, MINUS)
            tA = matrix_representation(qtA, PLUS) * scale
            tB = matrix_representation(qtB, MINUS) * scale

            C = np.zeros( (8,8) )
            C[0:4,  0: 4] = tA - tB
            C[0:4,  4: 8] = RB.T.dot(RA) - I
            C[4:8,  0: 4] = RB.T.dot(RA) - I
            C[4:8,  4: 8] = 0
            M += C.T.dot(C)

        [u,s,v] = linalg.svd(M)

        q,t = computeDualQuaternionNullSpace(v[-1,0:8], v[-2,0:8])

        E = matrix_representation(q, MINUS).T
        t = E.dot(t) / scale

        X = quaternion_matrix(q)
        X[0:3,3] = t[1:4]

        return (X, X)



