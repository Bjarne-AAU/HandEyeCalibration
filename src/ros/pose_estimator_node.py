import cv2

import numpy as np
import rospy

from slave_node import synchronized
from slave_node import SlaveNode
import conversions as convert

from nodes.display import ImageDisplay

import message_filters as mf

from sensor_msgs import point_cloud2 as pc2

from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import PointCloud2 as PointCloud2Msg
from sensor_msgs.msg import CameraInfo as CameraInfoMsg

from geometry_msgs.msg import PoseStamped as PoseMsg


from detector.detector import Detector
from estimator.pose_estimator import PoseEstimator

from tools.plugin_loader import PluginLoader

from hand_eye_calibration.msg import Command as CommandMsg
from hand_eye_calibration.msg import Reply as ReplyMsg



class PoseEstimatorNode(SlaveNode):

    def __init__(self, detector = "ArucoDetector", estimator="CloudPoseEstimator", camera="xtion"):
        super(PoseEstimatorNode, self).__init__("pose_estimator_node", "ESTIMATOR")

        self.__loader_detector = PluginLoader()
        self.__loader_detector.load("detector", Detector)

        self.__loader_estimator = PluginLoader()
        self.__loader_estimator.load("estimator", PoseEstimator)

        self._detector = None
        self._estimator = None
        self._camera = None
        self._pubImage = None
        self._pubMarker = None
        self._subCamera = None

        self.setDetector(detector)
        self.setEstimator(estimator)
        self.setCamera(camera)


    def __getParams(self, plugin):
        args = {}
        names, req, opt = PluginLoader.signature(plugin)
        args = opt.copy()
        for n in names:
            key = "~{}/{}".format(plugin.__name__, n)
            if rospy.has_param(key):
                args[n] = rospy.get_param(key)
        return args


    def setDetector(self, detector_name):
        try:
            detector_plugin = self.__loader_detector.getPluginByName(detector_name)
        except Exception as e:
            print("Detector " + detector_name + " not found!")
            return False

        args = self.__getParams(detector_plugin)
        self._detector = PluginLoader.instance(detector_plugin, args)
        if self._detector is None:
            print("Detector " + detector_name + " could not be instantiated!")
            return False

        if self._estimator is not None:
            self._estimator.setDetector(self._detector)
        return True


    def setEstimator(self, estimator_name):
        try:
            estimator_plugin = self.__loader_estimator.getPluginByName(estimator_name)
        except Exception as e:
            print("Estimator " + estimator_name + " not found!")
            return False

        args = self.__getParams(estimator_plugin)
        if self._detector is not None:
            args["detector"] = self._detector
        self._estimator = PluginLoader.instance(estimator_plugin, args)
        if self._estimator is None:
            print("Estimator " + estimator_name + " could not be instantiated!")
            return False
        return True


    def setCamera(self, camera):
        self._camera = camera
        if self.is_running:
            self.on_reset([], rospy.Time.now())



    def on_command(self, command, args, time):
        if command == "SET":
            what = args[0]

            if what == "DETECTOR":
                name = args[1]
                print("Select detector {}".format(name))
                if not self.setDetector(name):
                    self.send_error(command, args, "Setting detector failed")
                    return False

            elif what == "ESTIMATOR":
                name = args[1]
                print("Select estimator {}".format(name))
                if not self.setEstimator(name):
                    self.send_error(command, args, "Setting estimator failed")
                    return False

            elif what == "CAMERA":
                name = args[1]
                print("Select camera {}".format(name))
                self.setCamera(name)


    def on_start(self, args, time):
        print("Start estimator for {}".format(self._camera))

        topic = "/{}".format(self._camera)
        self._pubImage = rospy.Publisher(topic + "/marker_image", ImageMsg, queue_size=1)
        self._pubMarker = rospy.Publisher(topic + "/marker_pose", PoseMsg, queue_size=1)

        sub1 = mf.Subscriber(topic + "/rgb/camera_info", CameraInfoMsg, queue_size=3)
        sub2 = mf.Subscriber(topic + "/rgb/image_raw", ImageMsg, queue_size=3)
        sub3 = mf.Subscriber(topic + "/depth_registered/image_raw", ImageMsg, queue_size=3)
        sub4 = mf.Subscriber(topic + "/depth_registered/points", PointCloud2Msg, queue_size=3)
        self._subCamera = [sub1,sub2,sub3,sub4]
        self._sync = mf.ApproximateTimeSynchronizer(self._subCamera, 3, 0.01)
        self._sync.registerCallback(self.__callback)

    def on_stop(self, args, time):
        print("Stop estimator for {}".format(self._camera))

        for sub in self._subCamera:
            sub.unregister()
        self._subCamera = None
        self._sync = None

        self._pubImage.unregister()
        self._pubImage = None
        self._pubMarker.unregister()
        self._pubMarker = None

    def on_reset(self, args, time):
        if self.is_running:
            self.on_stop(args, time)
            self.on_start(args, time)


    @synchronized
    def __callback(self, cameraMsg, rgbMsg, depthMsg, cloudMsg):
        frame_id = rgbMsg.header.frame_id
        stamp = rgbMsg.header.stamp

        rgb = convert.image2matrix(rgbMsg, encoding='bgr8')
        depth = convert.image2matrix(depthMsg, encoding='passthrough').astype(np.float32)
        cloud = np.array(list(pc2.read_points(cloudMsg, field_names=["x","y","z"]))).reshape(cloudMsg.height, cloudMsg.width, -1)

        cameraMsg.K = np.array(cameraMsg.K, np.float32).reshape(3,3)
        cameraMsg.D = np.array(cameraMsg.D, np.float32)

        data = {}
        data["camera"] = cameraMsg
        data["rgb"] = rgb
        data["depth"] = depth
        data["cloud"] = cloud

        result = self._estimator.process(data)
        if result is None:
            markerPoseImage = convert.matrix2image(rgb, frame_id, stamp, encoding='bgr8')
            self._pubImage.publish(markerPoseImage)
            return
        frame, rvec, tvec = result

        markerPoseImage = convert.matrix2image(frame, frame_id, stamp, encoding='bgr8')
        self._pubImage.publish(markerPoseImage)

        markerPose = convert.axis2pose(rvec, tvec, frame_id, stamp)
        self._pubMarker.publish(markerPose)
