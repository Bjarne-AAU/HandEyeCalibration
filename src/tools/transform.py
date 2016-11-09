import numpy as np
import scipy.linalg as linalg


###################################
# Distances
###################################
def smallest_angle(T1, T2):
    q1 = quaternion_from_matrix(T1)
    q2 = quaternion_from_matrix(T2)

    k = 2 * q1.T.dot(q2)**2 - 1
    angle = np.arccos(np.clip(k, -1.0, 1.0))
    return np.rad2deg(angle.squeeze())

def smallest_dist(T1, T2):
    t1 = translation_from_transformation(T1)
    t2 = translation_from_transformation(T2)
    return linalg.norm(t1 - t2)

def smallest_error(T1, T2):
    angle = smallest_angle(T1, T2)
    dist = smallest_dist(T1, T2)
    return (angle, dist)


###################################
# Random
###################################
def random_vector(length=1):
    while True:
        v = np.random.random(3)*2-1
        vn = linalg.norm(v)
        if vn <= 1.0:
            break
    return length * v/vn

def random_rotation(max_angle, axis=None):
    if axis is None:
        axis = random_vector()
    axis /= linalg.norm(axis)
    max_angle = np.deg2rad(max_angle)
    r = np.random.random()**(1/3.0)
    return axis_angle_matrix(axis, r*max_angle)

def random_translation(max_dist, min_dist=0, axis=None):
    if axis is None:
        axis = random_vector()
    dist = min_dist + np.random.random()*(max_dist-min_dist)
    axis = axis/linalg.norm(axis)*dist
    return transformation_from_translation(axis)

def random_pose(max_angle, max_dist, min_dist=0, axis=None):
    TR = random_rotation(max_angle=max_angle)
    Tt = random_translation(min_dist=min_dist, max_dist=max_dist, axis=axis)
    return TR.dot(Tt)


###################################
# Transformation matrices
###################################
def transformation_from_rotation(R):
    T = np.identity(4)
    T[0:3,0:3] = R[0:3,0:3]
    return T

def transformation_from_translation(t):
    T = np.identity(4)
    T[0:3,3] = t.flatten()[0:3]
    return T

def transformation(R, t):
    T = np.identity(4)
    T[0:3,0:3] = R[0:3,0:3]
    T[0:3,3] = t.flatten()[0:3]
    return T

def translation_from_transformation(T):
    t = T[0:3,3:4]
    return t

def rotation_from_transformation(T):
    R = T[0:3,0:3]
    return R

def from_transformation(T):
    R = T[0:3,0:3]
    t = T[0:3,3:4]
    return (R,t)


def inv(T):
    TI = np.identity(4)
    RT = T[0:3,0:3].T
    TI[0:3,0:3] = RT
    TI[0:3,3:4] = -RT.dot(T[0:3,3:4])
    return TI



###################################
# Matrix representations for vectors (left/right crossproduct) and quaternions (left/right isoclinic)
PLUS = 1
MINUS = -1

def matrix_representation(v, sign = PLUS):
    v = v.ravel()
    if len(v) == 3:   return crossproduct_matrix_representation(v, sign)
    elif len(v) == 4: return quaternion_matrix_representation(v, sign)

def crossproduct_matrix_representation(v, sign = PLUS):
    skv = np.roll(np.roll(np.diag(v.ravel()), 1, 1), -1, 0)
    return sign*(skv - skv.T)

def quaternion_matrix_representation(q, sign = PLUS):
    M = np.identity(4) * q[0]
    M[1:4,0]   =  q[1:4]
    M[0,1:4]   = -q[1:4].T
    M[1:4,1:4] =  M[1:4,1:4] + sign*crossproduct_matrix_representation(q[1:4])
    return M




###################################
# Axis-Angle
###################################
def axis_angle_from_matrix(T):
    R = T[0:3,0:3]
    v = np.array((R - R.T)[[2,0,1], [1,2,0]])
    vn = linalg.norm(v)
    sina = 0.5*vn
    cosa = 0.5*(np.trace(R)-1)
    theta = np.arctan2(sina, cosa)
    if not np.isclose(vn, 0.0):
        v /= vn
    return (v[:, np.newaxis], theta)

def axis_angle_matrix(axis, theta):
    K = matrix_representation(axis)
    R = np.identity(3) + np.sin(theta) * K + (1-np.cos(theta)) * K.dot(K)
    return transformation_from_rotation(R)


###################################
# Quaternion
###################################
def quaternion_from_matrix(T):
    v,theta = axis_angle_from_matrix(T)
    return quaternion(np.cos(theta/2.0), np.sin(theta/2.0)*v)

def quaternion_matrix(q):
    q /= linalg.norm(q)
    QL = matrix_representation(q, PLUS)
    QR = matrix_representation(q, MINUS)
    R = QL.dot(QR.T)[1:4,1:4]
    return transformation_from_rotation(R)

def quaternion(qw, qv):
    q = np.hstack(( qw, qv.ravel()[0:3]))
    return q[:,np.newaxis]

def quaternion_from_imaginary(qv):
    q = np.hstack(( 0, qv.ravel()[0:3]))
    return q[:, np.newaxis]

def qinv(q):
    np.negative(q[1:4], q[1:4])
    return q[:, np.newaxis]

def qmult(p,q):
    return matrix_representation(p).dot(q)



###################################
# Dual-Quaternion
###################################
def dual_quaternion_from_matrix(T):
    R = T[0:3, 0:3]
    t = T[0:3, 3:4]
    qr = quaternion_from_matrix(R)
    qt = quaternion_from_imaginary(t)
    qt = 0.5 * qmult(qt, qr)
    return (qr, qt)

def dual_quaternion_matrix(dq):
    (qr, qt) = dq
    T = quaternion_matrix(qr)
    T[0:3, 3] = 2.0 * qmult(qt, qinv(qr))[1:4]
    return T

###################################
# Kronecker product
###################################
def vector_from_matrix(T):
    R = T[0:3, 0:3]
    return R.ravel()[:,np.newaxis]

def vector_matrix(v):
    R = np.reshape(v[0:9], (3,3))
    return transformation_from_rotation(R)


###################################
# Rotation matrices
###################################
def orthonormalize_rotation(T):
    R = T[0:3, 0:3]
    R = np.sign(linalg.det(R)) / pow(abs(linalg.det(R)), 1.0/3) * R
    [u,s,v] = linalg.svd(R)
    s = np.diag([1, 1, linalg.det(np.dot(u, v))])
    T[0:3, 0:3] = u.dot(s.dot(v))
    return T


###################################
# Quaternion
###################################
def computeDualQuaternionNullSpace(base1, base2):
    r1 = base1[0:4]
    v1 = base1[4:8]
    r2 = base2[0:4]
    v2 = base2[4:8]

    if np.isclose(r1.dot(v1), 0.0):
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

