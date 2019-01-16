import sys
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
            #f_string += '%g %g %g ' % (f[0], f[1], f[2])
            f_string += '{:>10.2f} {:>10.3f} {:>10.3f} '.format(f[0], f[1], f[2])
        f_string = f_string[:-1] + '\n'
        self._out.write(f_string)
        sys.stdout.flush()

class SubsetForceReporter(object):
    def __init__(self, filename, reportInterval, n_dim_keep):
        """Saves subset of system forces
        
        Useful if ignoring water molecules
        """
        self._out = open(filename, 'w')
        self._reportInterval = reportInterval
        self.n_dim_keep

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, True, False, None)

    def report(self, simulation, state):
        forces = state.getForces().value_in_unit(unit.kilojoules/unit.mole/unit.nanometer)
        f_string = ""
        for i in range(self.n_dim_keep):
            #f_string += '%g %g %g ' % (f[0], f[1], f[2])
            f_string += '{:>10.2f} {:>10.3f} {:>10.3f} '.format(forces[i][0], forces[i][1], forces[i][2])
        f_string = f_string[:-1] + '\n'
        self._out.write(f_string)
        sys.stdout.flush()


class MappedForceReporter(object):
    def __init__(self, filename, reportInterval, M):
        """Saves a linear combination of system forces
        
        When coarse-graining by combining atoms into 'beads', such as the center of mass of an amino acid,
        the forces of the
        atoms get mapped to the center of mass. 
        """
        self._out = open(filename, 'w')
        self._reportInterval = reportInterval

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, True, False, None)

    def report(self, simulation, state):
        forces = state.getForces().value_in_unit(unit.kilojoules/unit.mole/unit.nanometer)
        mapped_forces = np.dot(M, np.array(reduce(lambda x, y: x + y, forces))).reshape(-1, 3)
        f_string = ""
        for f in mapped_forces:
            #f_string += '%g %g %g ' % (f[0], f[1], f[2])
            f_string += '{:>10.2f} {:>10.3f} {:>10.3f} '.format(f[0], f[1], f[2])
        f_string = f_string[:-1] + '\n'
        self._out.write(f_string)
        sys.stdout.flush()

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
            #v_string += '%g %g %g ' % (v[0], v[1], v[2])
            v_string += '{:>10.2f} {:>10.3f} {:>10.3f} '.format(v[0], v[1], v[2])
        v_string = v_string[:-1] + '\n'
        self._out.write(v_string)
        sys.stdout.flush()

