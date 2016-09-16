# /usr/bin/env python

import numpy as np
import scipy.linalg as linalg

from tools.tools import *

from solver.AXXB.SolverAXXB import SolverAXXB


class AB(SolverAXXB):

    @property
    def shortName(self):
        return "XAxisR"

    def solveRotation(self):
        N = self.N
        A = np.zeros( (N,3,3) )
        b = np.zeros( (N,3,1) )

        for i in range(N):
            AA = axis_angle_from_matrix(self.RA(i))[0]
            AB = axis_angle_from_matrix(self.RB(i))[0]

            RA = matrix_representation(AA, PLUS)
            RB = matrix_representation(AB, MINUS)

            A[i] = RA - RB
            b[i] = AA - AB

        A = np.reshape(A, (A.shape[0]*A.shape[1],-1))
        b = np.reshape(b, (b.shape[0]*b.shape[1],-1))
        x = linalg.lstsq(A, -b)[0]

        x_norm = linalg.norm(x)
        x /= x_norm
        theta = 2 * np.arctan(x_norm)
        X = angle_axis_matrix(theta, x)

        return (X, X)
