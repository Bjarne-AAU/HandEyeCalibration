# /usr/bin/env python

import sys

import numpy as np
import scipy.io as io
import scipy.linalg as linalg

import transformations as tf
import tools

def random_point_on_sphere(radius=1):
    while True:
        v = tf.random_vector(3) * 2 - 1
        if linalg.norm(v) <= 1.0:
            break
    return radius * v/linalg.norm(v)


def random_rotation(max_angle=5.0):
    v = random_point_on_sphere()
    angle = np.deg2rad(max_angle)

    a = angle * 2 - angle
    a = np.random.random()*a
    return tools.angle_axis_matrix(a, v)

def random_translation(min_dist, max_dist=None):
    if max_dist == None:
        max_dist = min_dist

    r = min_dist + np.random.random()*(max_dist-min_dist)
    v = random_point_on_sphere(r)
    return tf.translation_matrix(v)

def random_pose(max_angle, min_dist, max_dist=None):
    TR = random_rotation(max_angle)
    Tt = random_translation(min_dist, max_dist)
    return Tt.dot(TR)

def random_pose_along_axis(max_angle_axis, min_dist, max_dist=None, axis=np.array([0,1,0]), max_angle_pose=180):
    if max_dist is None:
        max_dist = min_dist
        min_dist = 0

    angle = np.deg2rad(max_angle_axis)
    angle = angle * 2 - angle

    v = random_point_on_sphere()
    M = tools.angle_axis_matrix(angle, v)
    T = transform_from_translation((min_dist + np.random.random()*(max_dist-min_dist))*axis[:])
    R = random_rotation(max_angle_pose)
    return M.dot(T).dot(R)


def transform_from_translation(v):
    return tf.translation_matrix(v)

def transform_from_rotation(R):
    T = np.identity(4)
    T[0:3, 0:3] = R
    return T



def create_random_rotation(max_angle=180.0, axis=[1, 0, 0, 0], rand_r=None):
    if rand_r is None: rand_r = tf.random_vector(3)
    else: max_angle = 180.0

    # rand_r = normalized_vector(rand_r)

    dist = 360.0
    best = axis
    max_count = 100000
    count = 0
    while count < max_count:
        q = tf.random_quaternion(rand_r)
        angle = tools.angle_between_quaternions(q, axis)
        if angle < dist:
            dist = angle
            best = q

        if dist <= max_angle: break
        rand_r = tf.random_vector(3)
        # rand_r = normalized_vector(tf.random_vector(3))
        count += 1

    if count >= max_count:
        print("No valid rotation found after " + str(count) + " iterations with angle " + str(dist))
    # else:
    #     print("Found constrained rotation after " + str(count) + " iterations with angle " + str(dist))

    return tf.quaternion_matrix(best)


def create_random_translation(distance=1.0, rand_t=None):
    if rand_t is None: rand_t = tf.random_vector(3) * 2 - 1
    else: distance = linalg.norm(rand_t)

    return tf.translation_matrix(rand_t*distance)


    # norm = linalg.norm(rand_t)

    # if not np.isclose(norm, distance): rand_t *= distance/norm
    # return tf.translation_matrix(rand_t)


def create_random_pose(maxR=180.0, maxT=1.0, rand_r=None, rand_t=None):
    R = create_random_rotation(max_angle=maxR, rand_r=rand_r)
    # T = tf.translation_matrix([maxT,0,0])
    T = create_random_translation(maxT, rand_t)
    M = tf.concatenate_matrices(R, T)
    return M

def add_noise(poses, maxR=15.0, maxT=0.01):
    N = poses.shape[0]
    out = np.zeros([N, 4, 4])
    for n in range(N):
        R = random_rotation(max_angle=maxR)
        T = create_random_translation(maxT)
        # dist = np.random.random()*maxT
        # T = create_random_pose(maxR, dist)
        out[n,:,:] = tf.concatenate_matrices(poses[n,:,:], T, R)
    return out


def load(filename):
    data = io.loadmat(filename)
    RC  = data['RC']  # Robot       - Camera      : unknown
    CM  = data['CM']  # Camera      - Marker      : observed
    CMG = data['CMG'] # Camera      - Marker      : ground truth
    RE  = data['RE']  # Robot       - Endeffector : observed
    EM  = data['EM']  # Endeffector - Marker      : unknown
    return (RE.shape[0], RC, CM, CMG, RE, EM)

def save(filename, RC, CM, CMG, RE, EM):
    io.savemat(filename, mdict={'RC': RC, 'CM': CM, 'CMG': CMG, 'RE': RE, 'EM': EM,}, oned_as='row')
    return

def create(samples, verbose=True):
    # R = create_random_pose(maxR=30, maxT=0)
    # T = create_random_translation(distance=1)
    # RC  = tf.concatenate_matrices(R, T)
    RC = create_random_pose(maxR=90, maxT=2.5)  # unknown camera pose wrt robot
    # R = create_random_rotation(max_angle=30)
    # RC  = tf.concatenate_matrices(RC, R)
    # RC = np.identity(4)
    # RC[0,3] = 1.0

    # R = create_random_pose(maxR=15, maxT=0)
    # T = create_random_translation(distance=0.1)
    # EM  = tf.concatenate_matrices(R, T)
    EM = create_random_pose(maxR=25, maxT=0.15)  # unknown marker pose wrt endeffector
    # EM = create_random_translation(distance=0.15)
    # EM = np.identity(4)
    # EM[0,3] = 0.5

    RE = np.zeros([samples, 4, 4])      # observed endeffector poses wrt robot
    CM = np.zeros([samples, 4, 4])      # observed marker poses wrt camera

    for n in range(samples):
        if verbose:
            sys.stdout.write('\rCreate ' + str(samples) + ' samples: ' + str(round(n*100.0/samples, 2)) + '%')
            sys.stdout.flush()
        T1 = create_random_pose(maxR=40, maxT=np.random.uniform(0.25,1.0))
        T2 = create_random_pose(maxR=40, maxT=np.random.uniform(0.25,1.0))

        RE[n,:,:] = tf.concatenate_matrices(T1, T2)
        CM[n,:,:] = tf.concatenate_matrices(linalg.inv(RC), RE[n,:,:], EM)

    if verbose: print('')

    return (RC, CM, RE, EM)

