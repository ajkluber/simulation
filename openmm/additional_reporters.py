import simtk.unit as unit
import simtk.openmm as omm
import simtk.openmm.app as app

class ForceReporter(object):
    def __init__(self, filename, reportInterval):
        self._out = open(filename, 'w')
        self._reportInterval = reportInterval

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, True, False, None)

    def report(self, simulation, state):
        forces = state.getForces().value_in_unit(unit.kilojoules/unit.mole/unit.nanometer)
        f_string = ""
        for f in forces:
            f_string += '%g %g %g ' % (f[0], f[1], f[2])
        f_string = f_string[:-1] + '\n'
        self._out.write(f_string)

class VelocityReporter(object):
    def __init__(self, filename, reportInterval):
        self._out = open(filename, 'w')
        self._reportInterval = reportInterval

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, True, False, False, None)

    def report(self, simulation, state):
        velocities = state.getVelocities().value_in_unit(unit.nanometer/unit.picosecond)
        v_string = ""
        for v in velocities:
            v_string += '%g %g %g ' % (v[0], v[1], v[2])
        v_string = v_string[:-1] + '\n'
        self._out.write(v_string)

