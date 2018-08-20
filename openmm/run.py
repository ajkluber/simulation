import os
import shutil
import numpy as np

import simtk.unit as unit
import simtk.openmm as omm
import simtk.openmm.app as app

import util
import additional_reporters

global energy_minimization_tol
energy_minimization_tol = unit.Quantity(value=10., unit=unit.kilojoule_per_mole)

def adaptively_find_best_pressure(target_volume, ff_filename, name, n_beads,
        cutoff, r_switch, refT=300, saveas="press_equil.pdb", save_forces=False, cuda=False, p0=4000.):
    """Adaptively change pressure to reach target volume (density)"""

    temperature = refT*unit.kelvin
    collision_rate = 1.0/unit.picosecond
    timestep = 0.002*unit.picosecond
    n_steps = 5000
    nsteps_out = 100
    pressure = p0*unit.atmosphere    # starting pressure

    dynamics = "Langevin"
    ensemble = "NPT"

    util.add_elements(18*unit.amu, 37*unit.amu)

    traj_idx = 1
    all_files_exist = lambda idx: np.all([os.path.exists(x) for x in util.output_filenames(name, idx)])
    while all_files_exist(traj_idx):
        traj_idx += 1

    # get initial configuration
    pdb = app.PDBFile(name + "_min.pdb")
    topology = pdb.topology
    positions = pdb.positions

    templates = util.template_dict(topology, n_beads)
    min_name, log_name, traj_name, lastframe_name = util.output_filenames(name, traj_idx)

    forcefield = app.ForceField(ff_filename)

    system = forcefield.createSystem(topology,
            nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=cutoff,
            ignoreExternalBonds=True, residueTemplates=templates)

    # set switching function on nonbonded forces
    for i in range(system.getNumForces()):
        force = system.getForce(i) 
        if force.__repr__().find("NonbondedForce") > -1:
            force.setUseSwitchingFunction(True)
            if r_switch == 0:
                raise IOError("Set switching distance")
            else:
                force.setSwitchingDistance(r_switch/unit.nanometer)
    if cuda:
        properties = {'DeviceIndex': '0'}
        platform = omm.Platform.getPlatformByName('CUDA') 

    barostat = omm.MonteCarloBarostat(pressure, temperature)
    baro_idx = system.addForce(barostat)

    integrator = omm.LangevinIntegrator(temperature, collision_rate, timestep)

    if cuda:
        simulation = app.Simulation(topology, system, integrator, platform, properties)
    else:
        simulation = app.Simulation(topology, system, integrator)

    simulation.context.setPositions(positions)
    simulation.minimizeEnergy(tolerance=energy_minimization_tol)

    simulation.reporters.append(app.DCDReporter(traj_name, nsteps_out))
    simulation.reporters.append(app.StateDataReporter(log_name, nsteps_out,
        step=True, potentialEnergy=True, kineticEnergy=True, temperature=True,
        density=True, volume=True))

    save_forces = False
    if save_forces:
        simulation.reporters.append(additional_reporters.ForceReporter(name + "_forces_{}.dat".format(traj_idx), nsteps_out))

    print "Target: ", target_volume, " (nm^3)"
    print "Step    Pressure   Volume   Factor: "
    all_P = []
    all_V = []
    for i in range(200):
        # run at this pressure a little
        simulation.step(n_steps)

        P = simulation.context.getParameter(barostat.Pressure())
        state = simulation.context.getState()
        box_volume = state.getPeriodicBoxVolume()/(unit.nanometer**3)
        all_P.append(P)
        all_V.append(box_volume)

        perc_err_V = np.abs((target_volume - box_volume)/target_volume)*100.
        factor = (box_volume/target_volume)
        if (i > 10):
            if perc_err_V <= 10:
                print "{:<5d} {:>10.2f} {:>10.2f} {:>5.7f}  DONE".format(i + 1, P, box_volume, factor)
                # save last frame with new box dimensions
                state = simulation.context.getState()
                topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
                simulation.reporters.append(app.PDBReporter(lastframe_name, 1))
                simulation.step(1)
                shutil.copy(lastframe_name, saveas)
                break

        # update pressure
        print "{:<5d} {:>10.2f} {:>10.2f} {:>5.7f}".format(i + 1, P, box_volume, factor)
        old_pressure = pressure
        pressure = factor*old_pressure
        simulation.context.setParameter(barostat.Pressure(), pressure)

    all_P = np.array(all_P)
    all_V = np.array(all_V)

    np.save("pressure_in_atm_vs_step.npy", all_P)
    np.save("volume_in_nm3_vs_step.npy", all_V)

def equilibrate_unitcell_volume(pressure, ff_filename, name, n_beads, T, cutoff, r_switch, saveas="vol_equil.pdb", cuda=False):
    """Adaptively change pressure to reach target volume (density)"""

    traj_idx = 1
    temperature = T*unit.kelvin
    collision_rate = 1.0/unit.picosecond
    timestep = 0.002*unit.picosecond
    n_steps = 10000
    nsteps_out = 100

    dynamics = "Langevin"
    ensemble = "NPT"

    # get initial configuration
    pdb = app.PDBFile(name + "_min.pdb")
    topology = pdb.topology
    positions = pdb.positions

    templates = util.template_dict(topology, n_beads)

    if cuda:
        properties = {'DeviceIndex': '0'}
        platform = omm.Platform.getPlatformByName('CUDA') 

    min_name = name + "_min_{}.pdb".format(traj_idx)
    log_name = name + "_{}.log".format(traj_idx)
    traj_name = name + "_traj_{}.dcd".format(traj_idx)
    lastframe_name = name + "_fin_{}.pdb".format(traj_idx)

    forcefield = app.ForceField(ff_filename)

    system = forcefield.createSystem(topology,
            nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=cutoff,
            ignoreExternalBonds=True, residueTemplates=templates)

    nb_force = system.getForce(0) # assume nonbonded interactions are first force
    nb_force.setUseSwitchingFunction(True)
    if r_switch == 0:
        raise IOError("Set switching distance")
    else:
        nb_force.setSwitchingDistance(r_switch/unit.nanometer)

    integrator = omm.LangevinIntegrator(temperature, collision_rate, timestep)

    system.addForce(omm.MonteCarloBarostat(pressure, temperature))

    if cuda:
        simulation = app.Simulation(topology, system, integrator, platform, properties)
    else:
        simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy(tolerance=energy_minimization_tol)

    simulation.reporters.append(app.DCDReporter(traj_name, nsteps_out))
    simulation.reporters.append(app.StateDataReporter(log_name, nsteps_out,
        step=True, potentialEnergy=True, kineticEnergy=True, temperature=True,
        density=True, volume=True))

    # equilibrate at this pressure
    simulation.step(n_steps)

    # make sure to save the periodic box dimension in case they changed.
    state = simulation.context.getState()
    topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    simulation.reporters.append(app.PDBReporter(lastframe_name, 1))
    simulation.step(1)

    shutil.copy(lastframe_name, saveas)

def production(topology, positions, ensemble, temperature, timestep,
        collision_rate, pressure, n_steps, nsteps_out, ff_filename,
        firstframe_name, log_name, traj_name, lastframe_name, cutoff,
        templates, n_equil_steps=1000, nonbondedMethod=app.CutoffPeriodic,
        use_switch=False, r_switch=0, minimize=False, cuda=False,
        gpu_idxs=False, more_reporters=[], dynamics="Langevin"): 

    # load forcefield from xml file
    forcefield = app.ForceField(ff_filename)

    system = forcefield.createSystem(topology,
            nonbondedMethod=nonbondedMethod, nonbondedCutoff=cutoff,
            ignoreExternalBonds=True, residueTemplates=templates)

    if use_switch:
        nb_force = system.getForce(0) # assume nonbonded interactions are first force
        nb_force.setUseSwitchingFunction(True)
        if r_switch == 0:
            raise IOError("Set switching distance")
        else:
            nb_force.setSwitchingDistance(r_switch/unit.nanometer)
            
    if ensemble == "NVE": 
        integrator = omm.VerletIntegrator(timestep)
    else:
        if dynamics == "Langevin":
            integrator = omm.LangevinIntegrator(temperature, collision_rate, timestep)
        elif dynamics == "Brownian":
            integrator = omm.BrownianIntegrator(temperature, collision_rate, timestep)
        else:
            raise IOError("dynamics must be Langevin or Brownian")
        if ensemble == "NPT":
            system.addForce(omm.MonteCarloBarostat(pressure, temperature))
    
    if cuda:
        platform = omm.Platform.getPlatformByName('CUDA') 
        if gpu_idxs:
            properties = {'DeviceIndex': gpu_idxs}
        else:
            properties = {'DeviceIndex': '0'}

        simulation = app.Simulation(topology, system, integrator, platform, properties)
    else:
        simulation = app.Simulation(topology, system, integrator)

    # set initial positions and box dimensions
    simulation.context.setPositions(positions)
    #simulation.context.setPeriodicBoxVectors()

    if minimize:
        simulation.minimizeEnergy(tolerance=energy_minimization_tol)

    # initial equilibration
    simulation.step(n_equil_steps)

    # save the first frame minimized
    simulation.reporters.append(app.PDBReporter(firstframe_name, 1))
    simulation.step(1)
    simulation.reporters.pop(0)

    # record coordinates
    simulation.reporters.append(app.DCDReporter(traj_name, nsteps_out))
    simulation.reporters.append(app.StateDataReporter(log_name, nsteps_out,
        step=True, potentialEnergy=True, kineticEnergy=True, temperature=True,
        density=True, volume=True))

    # add user-defined reporters for e.g. forces or velocities
    if len(more_reporters) > 0:
        for i in range(len(more_reporters)):
            simulation.reporters.append(more_reporters[i])

    # run simulation!
    simulation.step(n_steps)

    # make sure to save the periodic box dimension in case they changed.
    state = simulation.context.getState()
    topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    simulation.reporters.append(app.PDBReporter(lastframe_name, 1))
    simulation.step(1)


if __name__ == "__main__":
    # sandbox

    refT = 300
    saveas = "c50_press_equil.pdb"
    name = "c50"
    n_beads = 50
    cuda = True
    target_volume = 11.0**3

    cutoff = 0.9*unit.nanometers
    vdwCutoff = 0.9*unit.nanometers
    r_switch = 0.7*unit.nanometers

    ff_filename = "ff_c50.xml"

    temperature = refT*unit.kelvin
    collision_rate = 1.0/unit.picosecond
    timestep = 0.002*unit.picosecond
    n_steps = 5000
    nsteps_out = 100
    pressure = 4000*unit.atmosphere    # starting pressure

    minimize = True
    dynamics = "Langevin"
    ensemble = "NPT"

    util.add_elements(18*unit.amu, 37*unit.amu)

    traj_idx = 1
    all_files_exist = lambda idx: np.all([os.path.exists(x) for x in util.output_filenames(name, idx)])
    while all_files_exist(traj_idx):
        traj_idx += 1

    # get initial configuration
    pdb = app.PDBFile(name + "_min.pdb")
    topology = pdb.topology
    positions = pdb.positions

    templates = util.template_dict(topology, n_beads)
    min_name, log_name, traj_name, lastframe_name = util.output_filenames(name, traj_idx)

    forcefield = app.ForceField(ff_filename)

    system = forcefield.createSystem(topology,
            nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=cutoff,
            ignoreExternalBonds=True, residueTemplates=templates)

    # set switching function on nonbonded forces
    for i in range(system.getNumForces()):
        force = system.getForce(i) 
        if force.__repr__().find("NonbondedForce") > -1:
            force.setUseSwitchingFunction(True)
            if r_switch == 0:
                raise IOError("Set switching distance")
            else:
                force.setSwitchingDistance(r_switch/unit.nanometer)
    if cuda:
        properties = {'DeviceIndex': '0'}
        platform = omm.Platform.getPlatformByName('CUDA') 

    barostat = omm.MonteCarloBarostat(pressure, temperature)
    baro_idx = system.addForce(barostat)

    integrator = omm.LangevinIntegrator(temperature, collision_rate, timestep)

    if cuda:
        simulation = app.Simulation(topology, system, integrator, platform, properties)
    else:
        simulation = app.Simulation(topology, system, integrator)

    simulation.context.setPositions(positions)
    simulation.minimizeEnergy(tolerance=energy_minimization_tol)

    simulation.reporters.append(app.DCDReporter(traj_name, nsteps_out))
    simulation.reporters.append(app.StateDataReporter(log_name, nsteps_out,
        step=True, potentialEnergy=True, kineticEnergy=True, temperature=True,
        density=True, volume=True))

    save_forces = False
    if save_forces:
        simulation.reporters.append(additional_reporters.ForceReporter(name + "_forces_{}.dat".format(traj_idx), nsteps_out))

    print "Target: ", target_volume, " (nm^3)"
    print "Step    Pressure   Volume   Factor: "
    all_P = []
    all_V = []
    for i in range(200):
        # run at this pressure a little
        simulation.step(n_steps)

        P = simulation.context.getParameter(barostat.Pressure())
        state = simulation.context.getState()
        box_volume = state.getPeriodicBoxVolume()/(unit.nanometer**3)
        all_P.append(P)
        all_V.append(box_volume)

        perc_err_V = np.abs((target_volume - box_volume)/target_volume)*100.
        factor = (box_volume/target_volume)
        if (i > 10):
            if perc_err_V <= 10:
                print "{:<5d} {:>10.2f} {:>10.2f} {:>5.7f}  DONE".format(i + 1, P, box_volume, factor)
                # save last frame with new box dimensions
                state = simulation.context.getState()
                topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
                simulation.reporters.append(app.PDBReporter(lastframe_name, 1))
                simulation.step(1)
                shutil.copy(lastframe_name, saveas)
                break

        # update pressure
        print "{:<5d} {:>10.2f} {:>10.2f} {:>5.7f}".format(i + 1, P, box_volume, factor)
        old_pressure = pressure
        pressure = factor*old_pressure
        simulation.context.setParameter(barostat.Pressure(), pressure)

    all_P = np.array(all_P)
    all_V = np.array(all_V)

    np.save("pressure_in_atm_vs_step.npy", all_P)
    np.save("volume_in_nm3_vs_step.npy", all_V)
    
    #N = len(all_P) - 1
    #avgV = np.mean(all_V[N/2:-1]) 
    #stdV = np.std(all_V[N/2:-1]) 
    #avgP = np.mean(all_P[N/2:-1]) 
    #stdP = np.std(all_P[N/2:-1]) 

    #np.savetxt("avgV.dat", np.array([avgV, stdV]))
    #np.savetxt("pressure.dat", np.array([avgP, stdP]))
    #np.savetxt("temperature.dat", np.array([refT]))
