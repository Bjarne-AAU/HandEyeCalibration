# /usr/bin/env python

from nodes.camera import Camera
from nodes.display import ImageDisplay

from detector.aruco_detector import DetectorAruco

cam = Camera()
disp = ImageDisplay()
detector = DetectorAruco(marker_size = 0.04, camera_params = "camera.yaml")

while disp.isOpen():
    frame = cam.getFrame()
    marker = detector.process(frame)
    detector.drawMarker(frame, marker)
    disp.show(frame)

cam.close()
