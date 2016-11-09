# /usr/bin/env python

import numpy as np
import scipy.linalg as linalg

from tools.transform import *

from solver.AXYB.SolverAXYB import SolverAXYB

class AB(SolverAXYB):

    @property
    def shortName(self):
        return "YQuatRT"

    def solve(self):
        N = self.N
        M = np.zeros( (16,16) )
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

            C = np.zeros( (8,16) )
            C[0:4,  0: 4] = RA
            C[0:4,  4: 8] = 0
            C[0:4,  8:12] = RB
            C[0:4, 12:16] = 0

            C[4:8,  0: 4] = tA.dot(RA)
            C[4:8,  4: 8] = RA
            C[4:8,  8:12] = RB.dot(tB)
            C[4:8, 12:16] = RB

            M += C.T.dot(C)

        [u,s,v] = linalg.svd(M)

        q,t = computeDualQuaternionNullSpace(v[-1,0:8], v[-2,0:8])

        E = quaternion_matrix_representation(q, MINUS).T
        t = E.dot(t)/scale

        X = quaternion_matrix(q)
        X[0:3,3] = t[1:4]


        q,t = computeDualQuaternionNullSpace(v[-1,8:16], v[-2,8:16])

        E = quaternion_matrix_representation(q, MINUS).T
        t = E.dot(t)/scale

        Y = quaternion_matrix(q)
        Y[0:3,3] = t[1:4]

        return (X, Y)

