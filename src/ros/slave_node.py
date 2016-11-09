from multiprocessing import Lock

import rospy

from tools.enum import enum

import message_filters as mf

from hand_eye_calibration.msg import Command as CommandMsg
from hand_eye_calibration.msg import Reply as ReplyMsg

COMMAND = enum(PING=CommandMsg.PING,
               ENABLE=CommandMsg.ENABLE,
               DISABLE=CommandMsg.DISABLE,
               START=CommandMsg.START,
               STOP=CommandMsg.STOP,
               RESET=CommandMsg.RESET,
               CUSTOM=CommandMsg.CUSTOM)

REPLY = enum(PONG=ReplyMsg.PONG,
             ACK=ReplyMsg.ACK,
             SUCCESS=ReplyMsg.SUCCESS,
             ERROR=ReplyMsg.ERROR,
             CUSTOM=ReplyMsg.CUSTOM)


def synchronized(func):
    def f(self, *args, **kwargs):
        with self._mutex:
            if not self.is_running:
                return
            func(self, *args, **kwargs)
    return f

class SlaveNode(object):

    def __init__(self, name, type = None, anonymous=True):
        rospy.init_node(name, anonymous=anonymous)

        self._mutex = Lock()

        self._type = type

        self.__out = rospy.Publisher("calibration/slaves/reply", ReplyMsg, queue_size=10)
        self.__in = mf.Subscriber("calibration/master/command", CommandMsg)
        self.__in.registerCallback(self.__callback)

        self.__enabled = True
        self.__running = False

    def run(self):
        rospy.spin()

    @property
    def is_enabled(self):
        return self.__enabled

    @property
    def is_running(self):
        return self.__running

    def is_valid_command(self, command, args):
        if command not in COMMAND.values():
            return False
        if command == COMMAND.CUSTOM:
            return args
        return True


    def send(self, reply, command, args=[]):
        if reply not in REPLY.values():
            print("REPLY FAILED: Unknown reply: {}".format(reply))
            return

        if isinstance(command, str):
            args = [command] + args
            command = COMMAND.CUSTOM

        # print("REPLY: {}{}".format(REPLY.reverse[reply], args))

        replyMsg = ReplyMsg()
        replyMsg.header.stamp = rospy.Time.now()
        replyMsg.node = rospy.get_name()
        replyMsg.command = command
        replyMsg.reply = reply
        replyMsg.args = args

        self.__out.publish(replyMsg)


    def send_error(self, command, args, msg):
        self.send(REPLY.ERROR, command, args + [msg])

    def send_success(self, command, args):
        self.send(REPLY.SUCCESS, command, args)


    def __callback(self, command_msg):
        node = rospy.get_name()
        stamp = command_msg.header.stamp
        nodes = command_msg.nodes
        command = command_msg.command
        args = command_msg.args

        if nodes and (node not in nodes and self._type not in nodes):
            return

        # if command in COMMAND.values():
        #     print("RECV: {}{}".format(COMMAND.reverse[command], args))
        # else:
        #     print("RECV: CMD({}) with args {}".format(command, args))

        if not self.is_valid_command(command, args):
            self.send_error(command, args, "Unknown or invalid command")
            return


        if command == COMMAND.PING:
            self.send(REPLY.PONG, command, [str(stamp.secs), str(stamp.nsecs), self._type])
            return

        elif command == COMMAND.ENABLE:
            if self.__enabled:
                self.send(REPLY.ACK, command, args, "Already enabled")
                return
            self.__enabled = True

        elif command == COMMAND.DISABLE:
            if not self.__enabled:
                self.send(REPLY.ACK, command, args + ["Already disabled"])
                return
            elif self.__running:
                self.send_error(command, args + ["Node is running"])
                return
            self.__enabled = False

        elif command == COMMAND.START:
            if not self.__enabled:
                self.send_error(command, args, "Node is disabled")
                return
            elif self.__running:
                self.send(REPLY.ACK, command, args + ["Already running"])
                return
            self.__running = True

        elif command == COMMAND.STOP:
            if not self.__enabled:
                self.send_error(command, args, "Node is disabled")
                return
            elif not self.__running:
                self.send(REPLY.ACK, command, args + ["Not running"])
                return
            self.__running = False

        elif command == COMMAND.RESET:
            if not self.__enabled:
                self.send_error(command, args, "Node is disabled")
                return

        elif command == COMMAND.CUSTOM:
            if not self.__enabled:
                self.send_error(command, args, "Node is disabled")
                return

        self.send(REPLY.ACK, command, args)

        success = False
        with self._mutex:
            success = self.__on_command(command, args, stamp)

        if success: self.send_success(command, args)


    def __on_command(self, command, args, stamp):
        result = False
        if   command == COMMAND.ENABLE:  result = self.on_enable(args, stamp)
        elif command == COMMAND.DISABLE: result = self.on_disable(args, stamp)
        elif command == COMMAND.START:   result = self.on_start(args, stamp)
        elif command == COMMAND.STOP:    result = self.on_stop(args, stamp)
        elif command == COMMAND.RESET:   result = self.on_reset(args, stamp)
        elif command == COMMAND.CUSTOM:  result = self.on_command(args[0], args[1:], stamp)

        if result is None: return True
        return result



    def on_enable(self, args, time):
        pass

    def on_disable(self, args, time):
        pass

    def on_start(self, args, time):
        pass

    def on_stop(self, args, time):
        pass

    def on_reset(self, args, time):
        pass

    def on_command(self, command, args, time):
        pass
