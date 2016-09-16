# /usr/bin/env python

import numpy as np
import scipy.linalg as linalg

from tools.tools import *

from solver.AXYB.SolverAXYB import SolverAXYB


class ABPrime(SolverAXYB):

    @property
    def shortName(self):
        return "YQuatT'"

    def computeBiQuaternionNullSpace(self, r1, v1, r2, v2):
        if (np.isclose(r1.dot(v1), 0.0)):
            (r1, r2) = (r2, r1)
            (v1, v2) = (v2, v1)

        a = r1.dot(v1)
        b = r1.dot(v2) + r2.dot(v1)
        c = r2.dot(v2)

        k = abs(b**2 - 4*a*c)
        s1 = (-b + np.sqrt(k)) / (2.0*a)
        s2 = (-b - np.sqrt(k)) / (2.0*a)


        s = np.array([s1, s2])
        sx = s**2 * r1.dot(r1) + 2*s*r1.dot(r2) + r2.dot(r2)

        ind = np.argmax(sx)
        L2 = np.sqrt(1.0 / sx[ind])
        L1 = s[ind] * L2

        r = L1*r1 + L2*r2
        v = L1*v1 + L2*v2
        return (r,v)

    def solve(self):
        N = self.N
        I = np.identity(4)
        M = np.zeros( (12,12) )
        scale = 1.0

        for i in range(N):
            qA = quaternion_from_matrix(self.RA(i))
            qB = quaternion_from_matrix(self.RB(i))
            qtA = np.vstack(([0], self.tA(i)))
            qtB = np.vstack(([0], self.tB(i)))

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

        q1 = v[-1,0:4]
        t1 = v[-1,4:8]
        q2 = v[-2,0:4]
        t2 = v[-2,4:8]

        (q,t) = self.computeBiQuaternionNullSpace(q1, t1, q2, t2)

        E = quaternion_matrix_representation(q, MINUS).T
        t = E.dot(t) / scale

        Y = tf.quaternion_matrix(q)
        Y[0:3,3] = t[1:4]

        X = np.identity(4)

        return (X,Y)

