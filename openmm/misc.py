from __future__ import print_function, absolute_import
import os
import time
import argparse
import numpy as np

import simtk.unit as unit
import simtk.openmm as omm
import simtk.openmm.app as app

import simulation.openmm.util as util

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('name', type=str, help='Name.')
    parser.add_argument('traj_idx', type=int, help='Trajectory index.')
    parser.add_argument('ensemble', type=str, help='Statistical ensemble to sample.')
    parser.add_argument('--T', type=float, default=300, help='Temperature.')
    parser.add_argument('--n_steps', type=int, default=int(5e6), help='Number of steps.')
    args = parser.parse_args()

    name = args.name
    traj_idx = args.traj_idx
    ensemble = args.ensemble
    T = args.T
    n_steps = args.n_steps
    
    assert ensemble in ["NVE", "NVT", "NPT"], "Must choose ensemble from: NVE, NVT, NPT"

    n_beads = int(name[1:])

    # parameters for coarse-grain polymer are taken from:
    # Anthawale 2007
    sigma_ply = 0.373*unit.nanometer
    eps_ply = 0.13986*unit.kilocalorie_per_mole
    mass_ply = 37.*unit.amu
    r0 = 0.153*unit.nanometer 
    kb = 334720.*unit.kilojoule_per_mole/(unit.nanometer**2)
    theta0 = 111*unit.degree
    ka = 462.*unit.kilojoule_per_mole/(unit.degree**2)
    bonded_params = [r0, kb, theta0, ka]

    # LJ solvent parameters.
    sigma_slv = 0.3151*unit.nanometer
    eps_slv = 1*unit.kilojoule_per_mole
    mass_slv = 18.*unit.amu
    packing_fraction = 0.85

    # simulation parameters
    nsteps_out = 100

    cutoff = 0.9*unit.nanometers
    vdwCutoff = 0.9*unit.nanometers
    L = n_beads*r0
    box_edge = L + 2*cutoff
    #temperature = 300.*unit.kelvin
    temperature = T*unit.kelvin
    pressure = 1.*unit.atmosphere
    collision_rate = 1.0/unit.picosecond
    timestep = 0.002*unit.picosecond
    #ensemble = "NPT"

    # we define our coarse-grain beads as additional elements in OpenMM
    app.element.solvent = app.element.Element(201, "Solvent", "Sv", mass_slv)
    app.element.polymer = app.element.Element(200, "Polymer", "Pl", mass_ply)

    starttime = time.time()

    # get initial configuration
    if traj_idx == 1:
        print("starting new simulation from: " + name + "_min.pdb")
        pdb = app.PDBFile(name + "_min.pdb")
    else:
        if os.path.exists(name + "_fin_{}.pdb".format(traj_idx - 1)):
            print("extending from" + name + "_fin_{}.pdb".format(traj_idx - 1))
            pdb = app.PDBFile(name + "_fin_{}.pdb".format(traj_idx - 1))
        elif os.path.exists(name + "_traj_{}.dcd".format(traj_idx - 1)) and os.path.exists(name + "_min.pdb"):
            print("extending from final frame of " + name + "_traj_{}.pdb".format(traj_idx - 1))
            import mdtraj as md
            traj = md.load(name + "_traj_{}.dcd".format(traj_idx - 1), top=name + "_min.pdb")
            traj[-1].save_pdb(name + "_fin_{}.pdb".format(traj_idx - 1))
            pdb = app.PDBFile(name + "_fin_{}.pdb".format(traj_idx - 1))
        else:
            raise IOError("No structure to start from!")

    #pdb = app.PDBFile(starting_pdb)
    topology = pdb.topology
    positions = pdb.positions

    ff_filename = "ff_c{}.xml".format(n_beads)
    util.write_ff_file(n_beads, eps_slv, sigma_slv, mass_slv, eps_ply,
            sigma_ply, mass_ply, bonded_params, cutoff, saveas=ff_filename)

    # tell OpenMM which residues are which in the forcefield. Otherwise
    # OpenMM is thrown by all residues having matching sets of atoms. 
    templates = {}
    idx = 1
    for res in topology.residues():
        templates[res] = "PL" + str(idx)
        if idx >= n_beads:
            break
        idx += 1

    # load forcefield from xml file
    forcefield = app.ForceField(ff_filename)

    min_name = name + "_min_{}.pdb".format(traj_idx)
    log_name = name + "_{}.log".format(traj_idx)
    traj_name = name + "_traj_{}.dcd".format(traj_idx)
    lastframe_name = name + "_fin_{}.pdb".format(traj_idx)

    system = forcefield.createSystem(topology,
            nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=cutoff,
            ignoreExternalBonds=True, residueTemplates=templates)

    system = forcefield.createSystem(topology,
            nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=cutoff,
            ignoreExternalBonds=True)

    if ensemble == "NVE":
        integrator = omm.VerletIntegrator(timestep)
    else:
        integrator = omm.LangevinIntegrator(temperature, collision_rate, timestep)
        if ensemble == "NPT":
            system.addForce(omm.MonteCarloBarostat(pressure, temperature))

    # Run simulation
    print("running production...")
    util.production_run(topology, positions, system, integrator, n_steps,
        nsteps_out, min_name, traj_name, lastframe_name, log_name)
    os.chdir("..")

    stoptime = time.time()
    print("{} steps took {} min".format(n_steps, (stoptime - starttime)/60.))
    os.chdir("..")
