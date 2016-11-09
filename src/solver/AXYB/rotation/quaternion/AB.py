# /usr/bin/env python

import numpy as np
import scipy.linalg as linalg

from tools.transform import *

from solver.AXYB.SolverAXYB import SolverAXYB

class AB(SolverAXYB):

    @property
    def shortName(self):
        return "YQuatR"

    def solveRotation(self):
        N = self.N
        M = np.zeros( (8,8) )

        for i in range(N):
            QA = quaternion_from_matrix(self.RA(i))
            QB = quaternion_from_matrix(self.RB(i))

            RA = matrix_representation(QA, PLUS)
            RB = matrix_representation(QB, MINUS)

            C = np.zeros( (4,8) )
            C[0:4,0:4] = RA
            C[0:4,4:8] = RB
            M += C.T.dot(C)

        [u,s,v] = linalg.svd(M)

        X = quaternion_matrix(v[-1,0:4])
        Y = quaternion_matrix(v[-1,4:8])

        return (X, Y)

