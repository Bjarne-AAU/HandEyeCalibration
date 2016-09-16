# /usr/bin/env python

import numpy as np
import scipy.linalg as linalg
import logger as log

import transformations as tf

PLUS = 1
MINUS = -1

def matrix_representation(v, sign = PLUS):
    v = v.flatten()
    if len(v) == 3:   return skew(v, sign)
    elif len(v) == 4: return quaternion_matrix_representation(v, sign)

def skew(v, sign = PLUS):
    skv = np.roll(np.roll(np.diag(v.flatten()), 1, 1), -1, 0)
    return sign*(skv - skv.T)

def quaternion_matrix_representation(Q, sign = PLUS):
    if Q.ndim == 1: Q = np.expand_dims(Q, axis=1)
    M = np.identity(4) * Q[0]
    M[1:4,0:1] =  Q[1:4]
    M[0:1,1:4] = -Q[1:4].T
    M[1:4,1:4] =  M[1:4,1:4] + sign*skew(Q[1:4])
    return M

def quaternion_from_matrix(R):
    q = tf.quaternion_from_matrix(R)
    return q[:, np.newaxis]

def quaternion_matrix(q):
    R = tf.quaternion_matrix(q)
    return R

def inv(T):
    TI = np.identity(4)
    RT = T[0:3,0:3].T
    TI[0:3,0:3] = RT
    TI[0:3,3:4] = -RT.dot(T[0:3,3:4])
    return TI

def orthonormalize_rotation(R):
    R = R[0:3, 0:3]
    R = np.sign(linalg.det(R)) / pow(abs(linalg.det(R)), 1.0/3) * R

    [u,s,v] = linalg.svd(R)
    s = np.diag([1, 1, linalg.det(np.dot(u, v))])
    return u.dot(s.dot(v))


def rotation_vector_from_matrix(T):
    # Exponential representation of rotation: log(R) = w
    R = T[0:3, 0:3]
    logR = linalg.logm(R).real
    return np.array([[logR[2,1]], [logR[0,2]], [logR[1,0]]])

def rotation_vector_matrix(w):
    # Exponential representation of rotation: R = exp(w)
    theta = linalg.norm(w)
    return angle_axis_matrix(theta, w/theta)

def axis_angle_from_matrix(T):
    w = rotation_vector_from_matrix(T)
    theta = linalg.norm(w)
    if theta != 0.0:
        w /= theta
    return (w, theta)

def angle_axis_from_matrix(T):
    # Exponential representation of rotation: R = exp(w) = exp(theta * K)
    # logm(R) = w = theta * K
    w = rotation_vector_from_matrix(T)
    theta = linalg.norm(w)
    if theta != 0.0:
        w /= theta
    return (theta, w)

def angle_axis_matrix(theta, axis):
    # using rodriguez rotation formula: R = exp(theta * K) = I + sin(theta)*K + (1-cos(theta))*K^2
    #                                                      = cos(theta)*I + sin(theta)*K + (1-cos(theta))*K.dot(K.T)
    K = skew(axis)
    T = np.identity(4)
    T[0:3, 0:3] = np.identity(3) + np.sin(theta) * K + (1-np.cos(theta)) * K.dot(K)
    return T

def dual_quaternion_from_matrix(T):
    R = T[0:3, 0:3]
    t = T[0:3, 3:4]

    qr = tf.quaternion_from_matrix(R)
    qt = np.array([0, t[0], t[1], t[2]])
    qt = 0.5 * tf.quaternion_multiply(qt, qr)

    return (qr, qt)


def dual_quaternion_matrix(dq):
    (qr, qt) = dq

    T = tf.quaternion_matrix(qr)
    T[0:3, 3] = 2.0 * tf.quaternion_multiply(qt, tf.quaternion_conjugate(qr))[1:4]

    return T






def angle_between_quaternions(q1, q2):
    k = 2 * np.dot(q1, q2)**2 - 1
    angle = np.arccos(max(-1.0, min(k, 1.0)))
    return np.rad2deg(angle)

def angle_between_quaternions2(q1, q2):
    q = tf.quaternion_multiply(q1, tf.quaternion_conjugate(q2))
    angle = 2*np.arctan2(linalg.norm(tf.quaternion_imag(q)), tf.quaternion_real(q))
    angle = min(angle, 2*np.pi-angle)
    return np.rad2deg(angle)

def angle_between_quaternions3(q1, q2):
    q = tf.quaternion_multiply(q1, tf.quaternion_conjugate(q2))
    angle = 2*np.arctan2(linalg.norm(tf.quaternion_imag(q)), tf.quaternion_real(q))
    return np.rad2deg(angle)


def reprojectionError(AA, BB, X, Y):
    samples = AA.shape[0]
    error = 0
    for i in range(samples):
        p1 = AA[i,:3,:3].dot(X[0:3,3]) + AA[i,0:3,3]
        p2 = Y[:3,:3].dot(BB[i,0:3,3]) + Y[0:3,3]
        error += np.inner(p1-p2,p1-p2)
        # error += linalg.norm(p1-p2)
    return error/samples

def rotationError(GT, EST):
    Q1 = tf.quaternion_from_matrix(GT)
    Q2 = tf.quaternion_from_matrix(EST)

    absAngle = angle_between_quaternions(Q1, Q2)
    relAngle = absAngle/180*100
    return (absAngle, relAngle)

def translationError(GT, EST):
    absPos = linalg.norm(GT[:3,3].flatten() - EST[:3,3].flatten())

    absPosGT = linalg.norm(GT[:3,3])
    if    absPos == 0:   relPos = 0
    elif  absPosGT == 0: relPos = np.nan
    else:                relPos = absPos/absPosGT*100
    return(absPos, relPos)


def printError(prefix, GT, EST, error):
    absAngle, relAngle = rotationError(GT, EST)
    absPos, relPos = translationError(GT, EST)

    out = prefix + " Error [t|R]: " + \
                    "{:8.4f}".format(absPos) + "   " + "{:8.4f}".format(absAngle) + "  |  " +  \
                    "{:6.2f}".format(relPos) + "%  " + "{:6.2f}".format(relAngle) + "%  |  " + \
                    "Reprojection: {:6.2f}".format(error*1000)


    log.testError(relPos < 5 and relAngle < 5, out)


