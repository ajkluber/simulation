import os
import shutil
import numpy as np
import xml.etree.ElementTree as ET

import simtk.unit as unit
import simtk.openmm as omm
import simtk.openmm.app as app

import util
import additional_reporters

global energy_minimization_tol
energy_minimization_tol = unit.Quantity(value=10., unit=unit.kilojoule_per_mole)

def adaptively_find_best_pressure(target_volume, ff_filename, name, n_beads,
        cutoff, r_switch, refT, save_forces=False, cuda=False, p0=4000.):
    """Adaptively change pressure to reach target volume (density)"""

    temperature = refT*unit.kelvin
    collision_rate = 1.0/unit.picosecond
    timestep = 0.002*unit.picosecond
    n_steps = 5000
    nsteps_out = 100
    pressure = p0*unit.atmosphere    # starting pressure

    dynamics = "Langevin"
    ensemble = "NPT"

    traj_idx = 1
    all_files_exist = lambda idx: np.all([os.path.exists(x) for x in util.output_filenames(name, idx)])
    while all_files_exist(traj_idx):
        traj_idx += 1

    # get initial configuration
    pdb = app.PDBFile(name + "_min.pdb")
    topology = pdb.topology
    positions = pdb.positions

    templates = util.template_dict(topology, n_beads)
    min_name, log_name, traj_name, final_state_name = util.output_filenames(name, traj_idx)

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
            if perc_err_V <= 5:
                print "{:<5d} {:>10.2f} {:>10.2f} {:>5.7f}  DONE".format(i + 1, P, box_volume, factor)
                simulation.saveState("final_state.xml")
                state = simulation.context.getState()
                box_vecs = state.getPeriodicBoxVectors()
                box_dims_in_nm = np.array([ box_vecs[x][x]/unit.nanometer for x in range(3) ])
                np.save("box_dims_nm.npy", box_dims_in_nm)
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

def equilibrate_unitcell_volume(pressure, ff_filename, name, n_beads, refT, T,
        cutoff, r_switch, prev_state_file, cuda=False):
    """Adaptively change pressure to reach target volume (density)"""

    traj_idx = 1
    starting_temperature = refT*unit.kelvin

    collision_rate = 1.0/unit.picosecond
    timestep = 0.002*unit.picosecond
    n_steps = 1000
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

    integrator = omm.LangevinIntegrator(starting_temperature, collision_rate, timestep)

    barostat = omm.MonteCarloBarostat(pressure, starting_temperature)
    system.addForce(barostat)

    if cuda:
        simulation = app.Simulation(topology, system, integrator, platform, properties)
    else:
        simulation = app.Simulation(topology, system, integrator)

    simulation.loadState(prev_state_file)

    simulation.reporters.append(app.DCDReporter(traj_name, nsteps_out))
    simulation.reporters.append(app.StateDataReporter(log_name, nsteps_out,
        step=True, potentialEnergy=True, kineticEnergy=True, temperature=True,
        density=True, volume=True))

    simulation.reporters.append(additional_reporters.ForceReporter(name + "_forces_{}.dat".format(traj_idx), nsteps_out))

    currT = starting_temperature 
    DeltaT = (T - refT)/100.
    for i in range(100):
        # equilibrate at this temperature pressure
        simulation.step(n_steps)

        # updte temperature closer to desired temperature
        currT += DeltaT*unit.kelvin
        simulation.integrator.setTemperature(currT)
        simulation.context.setParameter(barostat.Temperature(), currT)

    simulation.step(n_steps)

    # save final state of simulation.
    # includes positions, vels, box vectors.
    simulation.saveState("final_state_npt.xml")

    # save state without pressure info for NVT sims
    tree = ET.parse("final_state_npt.xml")
    for elem in tree.getroot():
        if elem.tag == "Parameters":
            elem.attrib.pop("MonteCarloPressure")
            elem.attrib.pop("MonteCarloTemperature")
    tree.write("final_state_nvt.xml", xml_declaration=True)
    
    state = simulation.context.getState()
    box_vecs = state.getPeriodicBoxVectors()
    box_dims_in_nm = np.array([ box_vecs[x][x]/unit.nanometer for x in range(3) ])
    np.save("box_dims_nm.npy", box_dims_in_nm)

def production(topology, positions, ensemble, temperature, timestep,
        collision_rate, pressure, n_steps, nsteps_out, ff_filename,
        firstframe_name, log_name, traj_name, final_state_name, cutoff,
        templates, n_equil_steps=1000, nonbondedMethod=app.CutoffPeriodic,
        prev_state_name=None, use_switch=False, r_switch=0, minimize=False,
        cuda=False, gpu_idxs=False, more_reporters=[], dynamics="Langevin"): 

    # load forcefield from xml file
    forcefield = app.ForceField(ff_filename)

    system = forcefield.createSystem(topology,
            nonbondedMethod=nonbondedMethod, nonbondedCutoff=cutoff,
            ignoreExternalBonds=True, residueTemplates=templates)

    if use_switch:
        # set switching function on nonbonded forces
        for i in range(system.getNumForces()):
            force = system.getForce(i) 
            if force.__repr__().find("NonbondedForce") > -1:
                force.setUseSwitchingFunction(True)
                if r_switch == 0:
                    raise IOError("Set switching distance")
                else:
                    force.setSwitchingDistance(r_switch/unit.nanometer)
            
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

    if prev_state_name is None:
        # set initial positions and box dimensions
        simulation.context.setPositions(positions)
        #simulation.context.setPeriodicBoxVectors()
    else:
        simulation.loadState(prev_state_name)

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

    # save final state. positions, box vectors. 
    simulation.saveState(final_state_name)


if __name__ == "__main__":
    # sandbox
    pass
