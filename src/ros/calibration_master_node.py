import copy

import rospy

import actionlib
import actionlib_msgs

import message_filters as mf

from geometry_msgs.msg import PoseStamped as PoseMsg

import tf
import conversions as convert

from master_node import synchronized
from master_node import ALL, COMMAND, REPLY
from master_node import MasterNode

from hand_eye_calibration.msg import CalibrationAction
from hand_eye_calibration.msg import CalibrationFeedback
from hand_eye_calibration.msg import CalibrationResult
from actionlib_msgs.msg import GoalStatus


class CalibrationMasterNode(MasterNode):

    def __init__(self, name, anonymous=False):
        super(CalibrationMasterNode, self).__init__(name, anonymous=anonymous)

        self._replies = {}
        self._received = 0

        self._status = GoalStatus()
        self._result = CalibrationResult()

        self._server = actionlib.SimpleActionServer("/calibration/server", CalibrationAction, execute_cb=self._process, auto_start = False)
        self._server.register_preempt_callback(self._cancel)
        self._server.start()

    def _cancel(self):
        self.set_preempted("Calibration cancelled")


    def is_active(self):
        return self._status.status == GoalStatus.ACTIVE

    def has_succeeded(self):
        return self._status.status == GoalStatus.SUCCEEDED

    def set_result(self, result):
        self._result = result

    def get_result(self):
        return self._result

    def set_status(self, status, msg, terminate=False):
        if self.is_active():
            self._status.status = status
            self._status.text = msg
        if terminate:
            self.terminate_goal()

    def set_preempted(self, msg, terminate=False):
        self._server.current_goal.set_cancel_requested()
        self.set_status(GoalStatus.PREEMPTED, msg)

    def set_aborted(self, msg, terminate=False):
        self.set_status(GoalStatus.ABORTED, msg)

    def set_succeeded(self, msg, terminate=False):
        self.set_status(GoalStatus.SUCCEEDED, msg)

    def terminate_goal(self):
        result = self.get_result()
        goal = self._status
        if goal.status == GoalStatus.SUCCEEDED:
            print("GOAL SUCCEEDED: " + goal.text)
            self._server.set_succeeded(result, text=goal.text)
        elif goal.status == GoalStatus.PREEMPTED:
            print("GOAL PREEMPTED: " + goal.text)
            self._server.set_preempted(result, text=goal.text)
        elif goal.status == GoalStatus.ABORTED:
            print("GOAL ABORTED: " + goal.text)
            self._server.set_aborted(result, text=goal.text)
        elif goal.status == GoalStatus.REJECTED:
            print("GOAL REJECTED: " + goal.text)
            self._server.set_aborted(result, text=goal.text)
        elif goal.status == GoalStatus.RECALLED:
            print("GOAL RECALLED: " + goal.text)
            self._server.set_aborted(result, text=goal.text)
        elif goal.status == GoalStatus.LOST:
            print("GOAL LOST: " + goal.text)
            self._server.set_aborted(result, text=goal.text)
        else:
            print("GOAL UNKNOWN: " + goal.text)
            self._server.set_aborted(result, text=goal.text)



    def _process(self, goal):
        self._status = copy.deepcopy(self._server.current_goal.get_goal_status())
        self._result = copy.deepcopy(self._server.get_default_result())

        nodes = self.discovery()
        print("Available nodes:")
        for n,t in nodes.iteritems():
            print("  {} [{}]".format(t, n[1:]))

        estimators = [n for n,t in nodes.iteritems() if t == "ESTIMATOR"]
        calibrators = [n for n,t in nodes.iteritems() if t == "CALIBRATOR"]
        motions = [n for n,t in nodes.iteritems() if t == "MOTION"]

        if not estimators:
            self.set_aborted("No estimator node found", True)
            return
        if not calibrators:
            self.set_aborted("No calibrator node found", True)
            return
        if not motions:
            self.set_aborted("No motion node found", True)
            return

        estimator = [estimators[0]]
        calibrator = [calibrators[0]]
        motion = [motions[0]]
        all = estimator + calibrator + motions

        self.send_and_wait(all, COMMAND.STOP)
        self.send_and_wait(all, COMMAND.RESET)

        self.send_and_wait(all, COMMAND.CUSTOM, ["SET", "CAMERA", goal.camera])
        self.send_and_wait(estimator, COMMAND.CUSTOM, ["SET", "DETECTOR", goal.detector_type])
        self.send_and_wait(estimator, COMMAND.CUSTOM, ["SET", "ESTIMATOR", goal.estimator_type])
        self.send_and_wait(calibrator, COMMAND.CUSTOM, ["SET", "CALIBRATOR", goal.calibrator_type])
        self.send_and_wait(calibrator, COMMAND.CUSTOM, ["SET", "SOLVER"] + goal.solver_type.split("."))

        self.send_and_wait(all, COMMAND.START)

        self._received = 0
        topic = "/{}/pose".format(goal.camera)
        sub = rospy.Subscriber(topic, PoseMsg, self.__callback, queue_size=3)

        while self.is_active():
            if rospy.is_shutdown(): self.set_aborted("Node shutdown")
            rospy.sleep(0.1)

        if not self.has_succeeded():
            sub.unregister()

        self.send_and_wait(motion, COMMAND.STOP, duration=30.0)
        sub.unregister()
        self.send_and_wait(estimator + calibrator, COMMAND.STOP, duration=30.0)

        self.terminate_goal()

        pose = self.get_result().pose
        T = convert.pose2matrix(pose)
        p = [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
        q = [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w]
        a = tf.transformations.euler_from_quaternion(q)

        print("Camera     : {}".format(goal.camera))
        print("Parent     : {}".format(pose.header.frame_id))
        print("Position   : {} {} {}".format(*p))
        print("Orientation: {} {} {} {}".format(*q))
        print("Euler (rpy): {} {} {}".format(*a))
        print("Xacro      : <origin xyz=\"{} {} {}\" rpy=\"{} {} {}\" />".format(p[0], p[1], p[2], a[0], a[1], a[2]))




    @synchronized
    def __callback(self, pose):
        self._received += 1
        print("Received camera pose estimate {}".format(self._received))

        feedback = CalibrationFeedback()
        feedback.pose = pose
        self._server.publish_feedback(feedback)

        result = CalibrationResult()
        result.pose = pose
        self.set_result(result)

        if self._received > 20:
            self.set_succeeded("Calibration successful")


    def discovery(self, duration=1.0):
        replies = self.send_and_wait(ALL, COMMAND.PING, duration=duration)
        nodes = {n:args[2] for n,(reply,cmd,args) in replies.iteritems() if reply == REPLY.PONG}
        return nodes

    def send_and_wait(self, nodes, command, args=[], duration=1.0):
        self._replies = {}
        self.send(nodes, command, args)

        if not nodes:
            rospy.sleep(duration)
            return copy.deepcopy(self._replies)

        timeout = rospy.Time.now() + rospy.Duration(duration)
        while True:
            rospy.sleep(0.1)
            replies = [self._replies.has_key(node) and self._replies[node][0] in [REPLY.SUCCESS, REPLY.ERROR] and self._replies[node][1] == command for node in nodes]
            if rospy.Time.now() > timeout or all(replies):
                break

        return copy.deepcopy(self._replies)

    @synchronized
    def on_ping(self, node, time, reply, cmd, args):
        self._replies.update({node:(reply, cmd, args)})

    @synchronized
    def on_acknowledge(self, node, time, reply, cmd, args):
        if not self._server.is_active():
            return
        self._replies.update({node:(reply, cmd, args)})

    @synchronized
    def on_success(self, node, time, reply, cmd, args):
        self._replies.update({node:(reply, cmd, args)})

    @synchronized
    def on_error(self, node, time, reply, cmd, args):
        if not self._server.is_active():
            self.send([node], COMMAND.STOP)
        if cmd in COMMAND.values():
            print("Error occurred while calibrating: {} says {}{}".format(node, COMMAND.reverse[cmd], args))
        else:
            print("Error occurred while calibrating: {} says {}{}".format(node, cmd, args))

        self._replies.update({node:(reply, cmd, args)})
        self._server.set_aborted(text=args[-1])

    @synchronized
    def on_custom(self, node, time, reply, cmd, args):
        if not self._server.is_active():
            self.send([node], COMMAND.STOP)
            return
        self._replies.update({node:(reply, cmd, args)})

