import simtk.unit as unit
import simtk.openmm as omm
import simtk.openmm.app as app

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

