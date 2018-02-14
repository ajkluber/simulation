import simtk.unit as unit
import simtk.openmm as omm
import simtk.openmm.app as app

def production_run(topology, positions, system, integrator, n_steps,
        nsteps_out, min_name, traj_name, lastframe_name, log_name):
    """Run trajectory"""
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()

    simulation.reporters.append(app.PDBReporter(min_name, 1))
    simulation.step(1)
    simulation.reporters.pop(0)

    simulation.reporters.append(app.DCDReporter(traj_name, nsteps_out))
    simulation.reporters.append(app.StateDataReporter(log_name, nsteps_out,
        step=True, potentialEnergy=True, kineticEnergy=True, temperature=True,
        density=True, volume=True))
    simulation.step(n_steps)

    simulation.reporters.append(app.PDBReporter(lastframe_name, 1))
    simulation.step(1)
