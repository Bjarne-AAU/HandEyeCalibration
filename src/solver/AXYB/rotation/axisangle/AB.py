# /usr/bin/env python

import numpy as np
import scipy.linalg as linalg

from tools.transform import *

from solver.AXYB.SolverAXYB import SolverAXYB

class AB(SolverAXYB):

    @property
    def shortName(self):
        return "YAxisR"

    def solveRotation(self):
        N = self.N
        I = np.identity(3)
        A = np.zeros( (N,3,6) )
        b = np.zeros( (N,3,1) )

        for i in range(N):
            QA = quaternion_from_matrix(self._AA[i, 0:3, 0:3])
            QB = quaternion_from_matrix(self._BB[i, 0:3, 0:3])

            k  = QA[0]/QB[0]
            AA = QA[1:4]/QA[0]
            AB = QB[1:4]/QB[0]

            A[i, 0:3, 0:3] = k*(I + matrix_representation(AA) + AA.dot(AA.T))
            A[i, 0:3, 3:6] =  -(I - matrix_representation(AB) + AA.dot(AB.T))

            b[i, 0:3] = (AB - AA)

        A = np.reshape(A, (A.shape[0]*A.shape[1],-1))
        b = np.reshape(b, (b.shape[0]*b.shape[1],-1))
        w = linalg.lstsq(A, b)[0]

        # angle axis form
        nx = linalg.norm(w[0:3])
        ny = linalg.norm(w[3:6])
        xv = w[0:3]/nx
        yv = w[3:6]/ny
        ya = 2*np.arctan(ny)
        xa = 2*np.arcsin(nx*np.cos(ya/2))

        X = axis_angle_matrix(xv, xa)
        Y = axis_angle_matrix(yv, ya)

        # quaternion form
        # yy = quaternion(1, w[3:6])
        # xx = quaternion(0, w[0:3])
        # y = yy / linalg.norm(yy)
        # x = xx / linalg.norm(yy)

        # x[0] = np.cos(np.arcsin(linalg.norm(x)))
        # # tmp = 1 - np.sum(x**2)
        # # sign = np.sign(tmp)
        # # x[0] = sign*np.sqrt(sign*tmp)

        # X = quaternion_matrix(x)
        # Y = quaternion_matrix(y)

        return (X, Y)
