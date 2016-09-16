# /usr/bin/env python

import importlib
import pkgutil
import sys
import inspect

import tools.tools as tools

from solver.SolverInterface import SolverInterface

def enum(**enums):
    return type('Enum', (), enums)

TYPE           = enum(ALL=None, AXXB="AXXB", AXYB="AXYB")
METHOD         = enum(ALL=None, R="rotation", T="translation", RT="rototranslation")
REPRESENTATION = enum(ALL=None, AXIS="axisangle", QUAT="quaternion", KRON="kronecker")


class SolverPlugins(object):

    def __init__(self):
        self._plugins = self.__import_plugins("solver", SolverInterface)

    def __import_plugins(self, package, base):
        results = []

        if isinstance(package, str):
            package = importlib.import_module(package)

        for loader, modname, is_pkg in pkgutil.walk_packages(path=package.__path__,
                                                             prefix=package.__name__+'.',
                                                             onerror=lambda x: None):
            if is_pkg: continue

            if modname not in sys.modules:
                importlib.import_module(modname)

            module = sys.modules[modname]

            pred = lambda member: inspect.isclass(member) and \
                                  member.__module__ == modname and \
                                  issubclass(member, base) and \
                                  not member.__subclasses__()

            classes = dict(inspect.getmembers(module, pred))
            results.extend(classes.values())

        results = [c for c in results if not c.__subclasses__()]

        return results

    def filter(self, name=None, type=None, method=None, representation=None):
        plugins = self._plugins
        if name is not None:
            if name.endswith("*"):
                plugins = [p for p in plugins if p.__name__.startswith(name[0:-1])]
            else:
                plugins = [p for p in plugins if name == p.__name__]
        if type is not None:
            plugins = [p for p in plugins if type == p.__module__.split(".")[1]]
        if method is not None:
            plugins = [p for p in plugins if method == p.__module__.split(".")[2]]
        if representation is not None:
            plugins = [p for p in plugins if representation == p.__module__.split(".")[3]]
        return plugins

    def create(self, name=None, type=None, method=None, representation=None):
        plugins = self.filter(name, type, method, representation)
        return [p() for p in plugins]

    def list(self, name=None, type=None, method=None, representation=None):
        plugins = self.filter(name, type, method, representation)
        return [p.__name__ for p in plugins]

    def istype(self, instance, type):
        plugins = self.filter(type=type)
        return instance.__class__ in plugins

    def usemethod(self, instance, method):
        plugins = self.filter(method=method)
        return instance.__class__ in plugins

    def userepresentation(self, instance, representation):
        plugins = self.filter(representation=representation)
        return instance.__class__ in plugins

