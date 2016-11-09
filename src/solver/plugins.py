# /usr/bin/env python

from tools.plugin_loader import PluginLoader
from solver.SolverInterface import SolverInterface

from tools.enum import enum

TYPE           = enum(ALL=None, AXXB="AXXB", AXYB="AXYB")
METHOD         = enum(ALL=None, R="rotation", T="translation", RT="rototranslation")
REPRESENTATION = enum(ALL=None, AXIS="axisangle", QUAT="quaternion", KRON="kronecker")


class SolverPlugins(PluginLoader):

    @classmethod
    def __match(self, plugin, name=None, type=None, method=None, representation=None):
        clazz = self.split(plugin)
        if name           is not None and not re.match(name, clazz[-1]): return False
        if type           is not None and type != clazz[1]:             return False
        if method         is not None and method != clazz[2]:           return False
        if representation is not None and representation != clazz[3]:   return False
        return True

    @classmethod
    def is_type(self, plugin, type):
        return type == self.split(plugin)[1]

    @classmethod
    def is_method(self, plugin, method):
        return method == self.split(plugin)[2]

    @classmethod
    def is_representation(self, plugin, representation):
        return representation == self.split(plugin)[3]


    def load(self, folder = "solver"):
        super(SolverPlugins, self).load(folder, SolverInterface)

    def filter(self, name=None, type=None, method=None, representation=None):
        self._plugins = [p for p in self._plugins if self.__match(p, name, type, method, representation)]

    def exclude(self, name=None, type=None, method=None, representation=None):
        self._plugins = [p for p in self._plugins if not self.__match(p, name, type, method, representation)]

    def getPlugin(self, name=None, type=None, method=None, representation=None):
        p = self._filter([name, name, representation, method, type])
        if len(p) == 0: raise Exception("No plugin with name " + str(name) + " found!")
        elif len(p) > 1: print("WARNING: Multiple plugins with name " + str(name) + " found!\n" + str(p))
        return p[0]
