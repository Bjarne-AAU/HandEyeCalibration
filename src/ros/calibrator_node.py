import cv2

import numpy as np

from slave_node import synchronized
from slave_node import SlaveNode
import conversions as convert

import rospy
import tf

import message_filters as mf

from geometry_msgs.msg import PoseStamped as PoseMsg


from solver.plugins import SolverPlugins
from solver.plugins import TYPE
from solver.plugins import METHOD
from solver.plugins import REPRESENTATION


from calibrator.calibrator import Calibrator

from tools.plugin_loader import PluginLoader


class CalibratorNode(SlaveNode):

    def __init__(self, calibrator="EyeInHandCalibrator", name=None, type=TYPE.AXYB, method=METHOD.RT, representation=REPRESENTATION.QUAT, camera="xtion", robot_frame = "base_link", hand_frame = "palm"):
        super(CalibratorNode, self).__init__("calibrator_node", "CALIBRATOR")

        rospy.set_param("~robot_frame", robot_frame)
        rospy.set_param("~hand_frame", hand_frame)

        self._transformer = tf.TransformListener()

        self.__loader_solver = SolverPlugins()
        self.__loader_solver.load("solver")

        self.__loader_calibrator = PluginLoader()
        self.__loader_calibrator.load("calibrator", Calibrator)

        self._solver = None
        self._calibrator = None
        self._camera = None

        self.setSolver(name, type, method, representation)
        self.setCalibrator(calibrator)
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


    def setSolver(self, name=None, type=None, method=None, representation=None):
        try:
            solver_plugin = self.__loader_solver.getPlugin(name, type, method, representation)
        except Exception as e:
            print(e)
            print("Solver " + str(name) + " not found!")
            return False

        args = self.__getParams(solver_plugin)
        self._solver = PluginLoader.instance(solver_plugin, args)
        if self._solver is None:
            print("Solver " + name + " could not be instantiated!")
            return False

        if self._calibrator is not None:
            self._calibrator.setSolver(self._solver)
        return True


    def setCalibrator(self, name):
        try:
            calibrator_plugin = self.__loader_calibrator.getPluginByName(name)
        except Exception as e:
            print("Calibrator " + name + " not found!")
            return False

        args = self.__getParams(calibrator_plugin)
        if self._solver is not None:
            args["solver"] = self._solver
        self._calibrator = PluginLoader.instance(calibrator_plugin, args)
        if self._calibrator is None:
            print("Calibrator " + name + " could not be instantiated!")
            return False
        return True

    def setCamera(self, camera):
        self._camera = camera
        if self.is_running:
            self.on_reset([], rospy.Time.now())


    def on_command(self, command, args, time):
        if command == "SET":
            what = args[0]

            if what == "SOLVER":
                type = args[1]
                method = args[2]
                representation = args[3]
                name = args[4]
                print("Select solver {}.{}.{}.{}".format(type, method, representation, name))
                if not self.setSolver(name, type, method, representation):
                    self.send_error(command, args, "Setting solver failed")
                    return False

            elif what == "CALIBRATOR":
                name = args[1]
                print("Select calibrator {}".format(name))
                if not self.setCalibrator(name):
                    self.send_error(command, args, "Setting calibrator failed")
                    return False

            elif what == "CAMERA":
                name = args[1]
                print("Select camera {}".format(name))
                self.setCamera(name)


    def on_start(self, args, time):
        print("Start calibrator for {}".format(self._camera))

        topic = "/{}/pose".format(self._camera)
        self._pubCamera = rospy.Publisher(topic, PoseMsg, queue_size=1)

        topic = "/{}/marker_pose".format(self._camera)
        self._subMarker = rospy.Subscriber(topic, PoseMsg, self.__callback, queue_size=3)


    def on_stop(self, args, time):
        print("Stop calibrator for {}".format(self._camera))

        self._subMarker.unregister()
        self._subMarker = None

        self._pubCamera.unregister()
        self._pubCamera = None


    def on_reset(self, args, time):
        if self.is_running:
            self.on_stop(args, time)
            self.on_start(args, time)
        self._calibrator.reset()



    def _lookup_transform(self, source, target, time, wait=1.0):
        try:
            self._transformer.waitForTransform(source, target, time, rospy.Duration(wait))
        except:
            pass
        try:
            transform = self._transformer.lookupTransform(source, target, time)
        except tf.ExtrapolationException as e:
            transform = None
        except tf.Exception as e:
            self.send_error("TRANSFORM", [source, target], str(e))
            transform = None
        return transform


    @synchronized
    def __callback(self, markerPose):
        robot_frame = rospy.get_param("Calibrator/robot_frame", "base_link")
        hand_frame = rospy.get_param("Calibrator/hand_frame", "palm")
        camera_link_frame = self._camera + "_link"
        camera_rgb_frame = self._camera + "_rgb_optical_frame"

        frame_id = markerPose.header.frame_id
        stamp = markerPose.header.stamp

        transform = self._lookup_transform(robot_frame, hand_frame, stamp)
        if transform is None: return

        # armPose = convert.transform2pose(transform, robot_frame, stamp)

        self._calibrator.setReferenceFrame(robot_frame)
        self._calibrator.setSolutionFrame(hand_frame)

        A = convert.transform2matrix(transform)
        B = convert.pose2matrix(markerPose)

        if not self._calibrator.addSample(A, B): return
        T = self._calibrator.solve()
        if T is not None:
            transform = self._lookup_transform(camera_rgb_frame, camera_link_frame, stamp)
            if transform is None: return

            T = T.dot(convert.transform2matrix(transform))
            cameraPose = convert.matrix2pose(T, self._calibrator.getReferenceFrame(), stamp)
            self._pubCamera.publish(cameraPose)
