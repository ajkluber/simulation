import numpy as np

import simtk.unit as unit
import simtk.openmm as omm
import simtk.openmm.app as app

import util

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

    # set initial positions 
    simulation.context.setPositions(positions)

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

    simulation.reporters.append(app.PDBReporter(lastframe_name, 1))
    simulation.step(1)

def adaptively_find_best_pressure(target_volume, ff_filename, name, n_beads, cutoff, r_switch, refT=300):
    """Adaptively change pressure to reach target volume (density)"""

    traj_idx = 1
    temperature = refT*unit.kelvin
    collision_rate = 1.0/unit.picosecond
    timestep = 0.002*unit.picosecond
    n_steps = 1000
    nsteps_out = 100

    minimize = False
    dynamics = "Langevin"
    ensemble = "NPT"

    # get initial configuration
    pdb = app.PDBFile(name + "_min.pdb")
    topology = pdb.topology
    positions = pdb.positions

    templates = util.template_dict(topology, n_beads)

    properties = {'DeviceIndex': '0'}
    platform = omm.Platform.getPlatformByName('CUDA') 

    # run at this pressure then adjust.
    new_pressure = 4000*unit.atmosphere    # starting pressure
    all_P = []
    all_V = []
    for i in range(200):
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

        system.addForce(omm.MonteCarloBarostat(new_pressure, temperature))

        simulation = app.Simulation(topology, system, integrator, platform, properties)
        simulation.context.setPositions(positions)

        simulation.reporters.append(app.DCDReporter(traj_name, nsteps_out))
        simulation.reporters.append(app.StateDataReporter(log_name, nsteps_out,
            step=True, potentialEnergy=True, kineticEnergy=True, temperature=True,
            density=True, volume=True))

        #simulation.reporters.append(sop.additional_reporters.ForceReporter(name + "_forces_{}.dat".format(traj_idx), nsteps_out))

        # equilibrate at this pressure a little
        simulation.step(n_steps)

        simulation.reporters.append(app.PDBReporter(lastframe_name, 1))
        simulation.step(1)

        # update pressure
        box_volume = np.loadtxt(log_name, delimiter=",", usecols=(4,))
        all_P.append(new_pressure.value_in_unit(unit.atmosphere))
        all_V.append(box_volume[-1])

        system.removeForce(4)
        factor = (box_volume[-1]/target_volume)

        print i +1, new_pressure, box_volume[-1], factor

        old_pressure = new_pressure
        new_pressure = factor*old_pressure
        
        traj_idx += 1

    all_P = np.array(all_P)
    all_V = np.array(all_V)

    np.save("pressure_in_atm_vs_step.npy", all_P)
    np.save("volume_in_nm3_vs_step.npy", all_V)
    
    avgV = np.mean(all_V[200:]) 
    stdV = np.std(all_V[200:]) 
    avgP = np.mean(all_P[200:]) 
    stdP = np.std(all_P[200:]) 

    np.savetxt("avgV.dat", np.array([avgV, stdV]))
    np.savetxt("pressure.dat", np.array([avgP, stdP]))
    np.savetxt("temperature.dat", np.array([refT]))

def equilibrate_unitcell_volume(pressure, ff_filename, name, n_beads, T, cutoff, r_switch):
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

    simulation = app.Simulation(topology, system, integrator, platform, properties)
    simulation.context.setPositions(positions)

    simulation.reporters.append(app.DCDReporter(traj_name, nsteps_out))
    simulation.reporters.append(app.StateDataReporter(log_name, nsteps_out,
        step=True, potentialEnergy=True, kineticEnergy=True, temperature=True,
        density=True, volume=True))

    # equilibrate at this pressure
    simulation.step(n_steps)

    simulation.reporters.append(app.PDBReporter(lastframe_name, 1))
    simulation.step(1)

