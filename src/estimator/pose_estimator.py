import cv2
import numpy as np
import scipy.linalg as linalg


def readCameraParameters(filename):
    if filename is None: return None

    fs = cv2.FileStorage(filename, cv2.FileStorage_READ)
    if not fs.isOpened():
        raise IOError("Failed to open " + str(filename))
    data = {}
    data["image_width"] = fs.getNode("image_width").real()
    data["image_height"] = fs.getNode("image_height").real()
    data["camera_name"] = fs.getNode("camera_name").string()
    data["camera_matrix"] = fs.getNode("camera_matrix").mat()
    data["distortion_coefficients"] = fs.getNode("distortion_coefficients").mat()
    fs.release()
    return data

def writeCameraParameters(data, filename):
    if filename is None: return

    fs = cv2.FileStorage(filename, cv2.FileStorage_WRITE)
    if not fs.isOpened():
        raise IOError("Failed to open " + str(filename))
    for k,v in data.iteritems():
        fs.write(k, v)
    fs.release()


class MetaClassDetector(type):
    def __repr__(self):
        return self.__name__

class PoseEstimator(object):
    __metaclass__ = MetaClassDetector
    def __repr__(self):
        return self.name

    def __init__(self, detector, camera_params=None):
        self._detector = detector
        self._camera_params = readCameraParameters(camera_params)

    @property
    def name(self):
        return type(self).__name__

    @property
    def detector(self):
        return self._detector

    def setDetector(self, detector):
        self._detector = detector

    def drawCube(self, frame, model, rvec, tvec, K, D):
        axisX = model.axisX / linalg.norm(model.axisX)
        axisY = model.axisY / linalg.norm(model.axisY)
        axisZ = np.cross(axisX, axisY)

        height = 0.1
        n = model.shape.shape[0]
        points = np.zeros((n*2+1,3))
        points[0: n] = model.shape
        points[n:-1] = model.shape + axisZ * height
        points,_ = cv2.projectPoints(points, rvec, tvec, K, D)
        points = [tuple(np.round(p).astype(np.int32)) for p in np.squeeze(points)]

        cv2.line(frame, points[0], points[1], (0,200,0))
        cv2.line(frame, points[1], points[2], (0,200,0))
        cv2.line(frame, points[2], points[3], (0,200,0))
        cv2.line(frame, points[3], points[0], (0,200,0))

        cv2.line(frame, points[4], points[5], (0,200,0))
        cv2.line(frame, points[5], points[6], (0,200,0))
        cv2.line(frame, points[6], points[7], (0,200,0))
        cv2.line(frame, points[7], points[4], (0,200,0))

        cv2.line(frame, points[0], points[4], (0,200,0))
        cv2.line(frame, points[1], points[5], (0,200,0))
        cv2.line(frame, points[2], points[6], (0,200,0))
        cv2.line(frame, points[3], points[7], (0,200,0))
        return frame

    def drawAxes(self, frame, model, rvec, tvec, K, D):
        axisX = model.axisX / linalg.norm(model.axisX)
        axisY = model.axisY / linalg.norm(model.axisY)
        axisZ = np.cross(axisX, axisY)

        length = 0.1
        points = np.zeros((4,3))
        points[0] = model.center
        points[1] = model.center + axisX * length
        points[2] = model.center + axisY * length
        points[3] = model.center + axisZ * length

        points,_ = cv2.projectPoints(points, rvec, tvec, K, D)
        points = [tuple(np.round(p).astype(np.int32)) for p in np.squeeze(points)]

        cv2.line(frame, points[0], points[1], (0,0,255), 2)
        cv2.line(frame, points[0], points[2], (0,255,0), 2)
        cv2.line(frame, points[0], points[3], (255,0,0), 2)


    def process(self, data):
        frame = np.copy(data["rgb"])
        marker = self._detector.process(frame, draw=False)
        if marker is None: return None

        model = self.detector.model

        if self._camera_params is not None:
            data["camera"].K = self._camera_params["camera_matrix"]
            data["camera"].D = self._camera_params["distortion_coefficients"]

        res = self._estimate(model, marker, data)
        if res is None: return
        rvec, tvec = res

        self.drawAxes(frame, model, rvec, tvec, data["camera"].K, data["camera"].D)
        self.drawCube(frame, model, rvec, tvec, data["camera"].K, data["camera"].D)

        return frame, rvec, tvec


    def _estimate(self, model, marker, data):
        raise NotImplementedError()

