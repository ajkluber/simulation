import simtk.unit as unit
import simtk.openmm as omm
import simtk.openmm.app as app

def production(topology, positions, ensemble, temperature, timestep,
        collision_rate, pressure, n_steps, nsteps_out, ff_filename,
        firstframe_name, log_name, traj_name, lastframe_name, cutoff,
        templates, nonbondedMethod=app.CutoffPeriodic, minimize=False): 

    # load forcefield from xml file
    forcefield = app.ForceField(ff_filename)

    system = forcefield.createSystem(topology,
            nonbondedMethod=nonbondedMethod, nonbondedCutoff=cutoff,
            ignoreExternalBonds=True, residueTemplates=templates)
            
    if ensemble == "NVE": 
        integrator = omm.VerletIntegrator(timestep)
    else:
        integrator = omm.LangevinIntegrator(temperature, collision_rate, timestep)
        if ensemble == "NPT":
            system.addForce(omm.MonteCarloBarostat(pressure, temperature))
    
    ##if cuda:
    ##    platform = Platform.getPlatformByName('CUDA') 
    ##    if gpu_idxs:
    ##        properties = {'DeviceIndex': gpu_idxs}
    ##    else:
    ##        properties = {'DeviceIndex': '0'}

    #platform = omm.Platform.getPlatformByName('CUDA')
    #properties = {'DeviceIndex': '0'}

    # Run simulation
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)

    if minimize:
        simulation.minimizeEnergy()

    # save the first frame minimized
    simulation.reporters.append(app.PDBReporter(firstframe_name, 1))
    simulation.step(1)
    simulation.reporters.pop(0)

    simulation.reporters.append(app.DCDReporter(traj_name, nsteps_out))
    simulation.reporters.append(app.StateDataReporter(log_name, nsteps_out,
        step=True, potentialEnergy=True, kineticEnergy=True, temperature=True,
        density=True, volume=True))
    simulation.step(n_steps)

    simulation.reporters.append(app.PDBReporter(lastframe_name, 1))
    simulation.step(1)

