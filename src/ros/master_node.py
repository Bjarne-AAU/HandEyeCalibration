import sys
from multiprocessing import Lock

import rospy

import message_filters as mf

from tools.enum import enum

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

ALL = []

def synchronized(func):
    def f(self, *args, **kwargs):
        with self._mutex:
            func(self, *args, **kwargs)
    return f


class MasterNode(object):

    def __init__(self, name, anonymous=False):
        rospy.init_node(name, anonymous=anonymous)

        self._mutex = Lock()

        self.__out = rospy.Publisher("calibration/master/command", CommandMsg, queue_size=10)
        self.__in = mf.Subscriber("calibration/slaves/reply", ReplyMsg)
        self.__in.registerCallback(self.__callback)


    def send(self, nodes, command, args=[]):
        if command not in COMMAND.values():
            print("SEND FAILED: Unknown command: {}".format(command))
            return

        # print("SEND: {}{} to {}".format(COMMAND.reverse[command], args, nodes))
        # sys.stdout.flush()

        commandMsg = CommandMsg()
        commandMsg.header.stamp = rospy.Time.now()
        commandMsg.nodes = nodes
        commandMsg.command = command
        commandMsg.args = args

        self.__out.publish(commandMsg)


    def __callback(self, reply_msg):
        stamp = reply_msg.header.stamp
        node = reply_msg.node
        command = reply_msg.command
        reply = reply_msg.reply
        args = reply_msg.args

        if reply not in REPLY.values():
            print("RECV FAILED: Unknown reply {} from {}".format(reply, node))
            return

        # if command in COMMAND.values():
        #     print("{}: {}{} from [{}]".format(COMMAND.reverse[command], REPLY.reverse[reply], args, node))
        # else:
        #     print("CMD({}): {}{} from [{}]".format(command, REPLY.reverse[reply], args, node))
        # sys.stdout.flush()

        if reply == REPLY.PONG:     self.on_ping(node, stamp, reply, command, args)
        elif reply == REPLY.ACK:    self.on_acknowledge(node, stamp, reply, command, args)
        elif reply == REPLY.SUCCESS:self.on_success(node, stamp, reply, command, args)
        elif reply == REPLY.ERROR:  self.on_error(node, stamp, reply, command, args)
        elif reply == REPLY.CUSTOM: self.on_custom(node, stamp, reply, command, args)



    def on_ping(self, node, time, reply, cmd, args):
        ping = (time - rospy.Time(long(args[0]), long(args[1]))).to_sec()*1000.0
        pong = (rospy.Time.now() - time).to_sec()*1000.0
        print("[{}]   {}: {:6.2f} ms    {}: {:6.2f} ms".format(node, cmd, ping, REPLY.reverse[REPLY.PONG], pong))


    def on_acknowledge(self, node, time, reply, cmd, args):
        pass

    def on_success(self, node, time, reply, cmd, args):
        pass

    def on_error(self, node, time, reply, cmd, args):
        pass

    def on_custom(self, node, time, reply, cmd, args):
        pass

    def run(self):
        rospy.spin()
