import numpy as np
import scipy.linalg as linalg
import logger as log
import transform as tf


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
    absAngle = tf.smallest_angle(GT, EST)
    relAngle = absAngle/180*100
    return (absAngle, relAngle)

def translationError(GT, EST):
    absPos = tf.smallest_dist(GT, EST)

    absPosGT = linalg.norm(GT[:3,3])
    if    absPos == 0:   relPos = 0
    elif  absPosGT == 0: relPos = np.nan
    else:                relPos = absPos/absPosGT*100
    return(absPos, relPos)


def printError(prefix, absAngle, relAngle, absPos, relPos, error):
    out = prefix + " Error [t|R]: " + \
                    "{:8.4f}".format(absPos) + "   " + "{:8.4f}".format(absAngle) + "  |  " +  \
                    "{:6.2f}".format(relPos) + "%  " + "{:6.2f}".format(relAngle) + "%  |  " + \
                    "Reprojection: {:6.2f}".format(error*1000.0)

    log.testError(relPos < 5 and relAngle < 5, out)

