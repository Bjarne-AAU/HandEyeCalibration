import numpy as np
import copy

from Queue import Queue

from slave_node import synchronized
from slave_node import SlaveNode

import conversions as convert
import tools.transform as transform

import rospy
import tf

import moveit_commander
from moveit_msgs.msg import DisplayTrajectory as DisplayTrajectoryMsg

from geometry_msgs.msg import PoseStamped as PoseMsg


class ArmMotionNode(SlaveNode):

    def __init__(self):
        super(ArmMotionNode, self).__init__("arm_motion_node", "MOTION")

        group = rospy.get_param("~group", "arm")
        planner = rospy.get_param("~planner", "PRMkConfigDefault")
        attempts = 3
        time = 3.0

        print("Starting arm motion node with group " + group)

        self._pubMove = rospy.Publisher("/move_group/display_planned_path", DisplayTrajectoryMsg, queue_size=1)

        self._robot = moveit_commander.RobotCommander()

        self._arm = self._robot.get_group(group)
        # self._arm = moveit_commander.MoveGroupCommander(group)
        self._arm.set_planner_id(planner)
        self._arm.set_num_planning_attempts(attempts)
        self._arm.set_planning_time(time)

        self._planTimer = None
        self._execTimer = None
        self._lastExecDuration = rospy.Duration(0)

        self._queue = Queue(1)

        self._speed = 0.5
        self._max_angle = 25
        self._max_translation = [0.3, 0.3, 0.01]
        self._max_errors = 10
        self._done = True


    def _scale_trajectory(self, trajectory, scale):
        for p in trajectory.joint_trajectory.points:
            p.time_from_start /= scale
            p.velocities = np.array(p.velocities)*scale
            p.accelerations = np.array(p.accelerations)*scale
        return trajectory

    def _plan(self, event):
        if self._done: return
        self._arm.remember_joint_values("HOME")
        base_pose = convert.pose2matrix(self._arm.get_current_pose())
        home_counter = 0
        error_counter = 0

        previous_state = copy.deepcopy(self._robot.get_current_state())
        while True:
            if error_counter > self._max_errors:
                self.send_error("PLANNING", [], "Failed too often to plan to next pose")

            self._arm.set_start_state(previous_state)

            if not self._done or home_counter > 5:
                random_translation = transform.random_translation(1.0)
                random_translation[0,3] *= self._max_translation[0]
                random_translation[1,3] *= self._max_translation[1]
                random_translation[2,3] *= self._max_translation[2]

                random_rotation = transform.random_rotation(self._max_angle)
                new_pose = convert.matrix2pose(random_translation.dot(base_pose).dot(random_rotation))
                self._arm.set_pose_target(new_pose)
                home_counter = 0
            else:
                self._arm.set_named_target("HOME")
                home_counter += 1
                print("Planning home: {}".format(home_counter))

            # do the planning
            plan = self._arm.plan()

            # replan if trajectory constraints not satified
            points = len(plan.joint_trajectory.points)
            if points <= 0:
                error_counter += 1
                continue

            if points > 15:
                error_counter += 1
                print("Complex trajectory with {} points".format(points))
                continue

            error_counter = 0

            # scale trajectory speed
            self._scale_trajectory(plan, self._speed)

            # display plan
            trajectoryMsg = DisplayTrajectoryMsg()
            trajectoryMsg.trajectory_start = previous_state
            trajectoryMsg.trajectory = [plan]
            self._pubMove.publish(trajectoryMsg)


            # update robot state
            previous_state = copy.deepcopy(self._robot.get_current_state())
            previous_state.joint_state.position = list(previous_state.joint_state.position)
            for i, joint in enumerate(plan.joint_trajectory.joint_names):
                idx = previous_state.joint_state.name.index(joint)
                previous_state.joint_state.position[idx] = plan.joint_trajectory.points[-1].positions[i]

            # add plan
            self._queue.put(plan)

            # stop after plan home found
            if self._done and home_counter > 0:
                break

        self._arm.forget_joint_values("HOME")



    def _exec(self, event):
        exec_counter = 0
        while not self._done or not self._queue.empty():
            exec_counter += 1
            plan = self._queue.get()
            duration = plan.joint_trajectory.points[-1].time_from_start
            success = False
            while not success:
                success = self._arm.execute(plan, False)
                rospy.sleep(0.1)

            rospy.sleep(duration)
            self._queue.task_done()



    def on_start(self, args, time):
        print("Start motion for {}".format(self._arm.get_name()))
        self._done = False

        self._planTimer = rospy.Timer(rospy.Duration(0.1), self._plan, oneshot=True)
        self._execTimer = rospy.Timer(rospy.Duration(0.1), self._exec, oneshot=True)


    def on_stop(self, args, time):
        self._done = True
        self._queue.join()
        print("Stop arm motion for {}".format(self._arm.get_name()))


    def on_reset(self, args, time):
        if self.is_running:
            self.on_stop(args, time)
            self.on_start(args, time)

