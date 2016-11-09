from tools import transform

from calibrator import Calibrator

from solver.plugins import SolverPlugins
from solver.plugins import TYPE


class EyeOnHandCalibrator(Calibrator):

    def getReferenceFrame(self):
        return self._frame_reference

    def getSolutionFrame(self):
        return self._frame_solution

    def _prepareSample(self, oldA, A, oldB, B):
        newA = None
        newB = None

        if SolverPlugins.is_type(self._solver, TYPE.AXXB):
            if oldA is not None and oldB is not None:
                newA = oldA.dot(transform.inv(A))
                newB = oldB.dot(transform.inv(B))
        elif SolverPlugins.is_type(self._solver, TYPE.AXYB):
            newA = A
            newB = B
        else:
            raise("Unknown solver")

        return (newA, newB)
