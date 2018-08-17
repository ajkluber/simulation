import os
import numpy as np

import simtk.unit as unit
import simtk.openmm as omm
import simtk.openmm.app as app

import util
import additional_reporters

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
        simulation.minimizeEnergy()

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

def adaptively_find_best_pressure(target_volume, ff_filename, name, n_beads, cutoff, r_switch, refT=300, save_forces=False, cuda=False):
    """Adaptively change pressure to reach target volume (density)"""

    temperature = refT*unit.kelvin
    collision_rate = 1.0/unit.picosecond
    timestep = 0.002*unit.picosecond
    n_steps = 5000
    nsteps_out = 100

    minimize = False
    dynamics = "Langevin"
    ensemble = "NPT"

    traj_idx = 1
    all_files_exist = lambda idx: np.all([os.path.exists(x) for x in util.output_filenames(name, idx)])
    while all_files_exist(traj_idx):
        traj_idx += 1

    # get initial configuration

    if cuda:
        properties = {'DeviceIndex': '0'}
        platform = omm.Platform.getPlatformByName('CUDA') 

    # run at this pressure then adjust.
    new_pressure = 4000*unit.atmosphere    # starting pressure
    all_P = []
    all_V = []
    for i in range(200):
        if i == 0:
            pdb = app.PDBFile(name + "_min.pdb")
        else:
            pdb = app.PDBFile(name + "_fin_{}.pdb".format(traj_idx - 1))
        topology = pdb.topology
        positions = pdb.positions

        templates = util.template_dict(topology, n_beads)
        min_name, log_name, traj_name, lastframe_name = util.output_filenames(name, traj_idx)

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

        system.addForce(omm.MonteCarloBarostat(new_pressure, temperature))

        if cuda:
            simulation = app.Simulation(topology, system, integrator, platform, properties)
        else:
            simulation = app.Simulation(topology, system, integrator)
        simulation.context.setPositions(positions)

        simulation.reporters.append(app.DCDReporter(traj_name, nsteps_out))
        simulation.reporters.append(app.StateDataReporter(log_name, nsteps_out,
            step=True, potentialEnergy=True, kineticEnergy=True, temperature=True,
            density=True, volume=True))

        if save_forces:
            simulation.reporters.append(additional_reporters.ForceReporter(name + "_forces_{}.dat".format(traj_idx), nsteps_out))

        # equilibrate at this pressure a little
        simulation.step(n_steps)

        # make sure to save the periodic box dimension in case they changed.
        state = simulation.context.getState()
        topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
        simulation.reporters.append(app.PDBReporter(lastframe_name, 1))
        simulation.step(1)

        # update pressure
        box_volume = np.loadtxt(log_name, delimiter=",", usecols=(4,))
        all_P.append(new_pressure.value_in_unit(unit.atmosphere))
        all_V.append(box_volume[-1])

        perc_err = np.abs((target_volume - box_volume[-1])/target_volume)*100.
        if perc_err <= 1 and i > 10:
            break
        else:
            factor = (box_volume[-1]/target_volume)
            if box_volume[-1] > target_volume:
                factor = np.min([1.05, factor])
            else:
                factor = np.max([0.95, factor])

        system.removeForce(4)
        print traj_idx, new_pressure, box_volume[-1], factor

        old_pressure = new_pressure
        new_pressure = factor*old_pressure
        
        traj_idx += 1

    all_P = np.array(all_P)
    all_V = np.array(all_V)

    np.save("pressure_in_atm_vs_step.npy", all_P)
    np.save("volume_in_nm3_vs_step.npy", all_V)
    
    N = len(all_P)
    avgV = np.mean(all_V[N/2:]) 
    stdV = np.std(all_V[N/2:]) 
    avgP = np.mean(all_P[N/2:]) 
    stdP = np.std(all_P[N/2:]) 

    np.savetxt("avgV.dat", np.array([avgV, stdV]))
    np.savetxt("pressure.dat", np.array([avgP, stdP]))
    np.savetxt("temperature.dat", np.array([refT]))

def equilibrate_unitcell_volume(pressure, ff_filename, name, n_beads, T, cutoff, r_switch, cuda=False):
    """Adaptively change pressure to reach target volume (density)"""

    traj_idx = 1
    temperature = T*unit.kelvin
    collision_rate = 1.0/unit.picosecond
    timestep = 0.002*unit.picosecond
    n_steps = 10000
    nsteps_out = 100

    minimize = False
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

