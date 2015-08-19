import os
import shutil
import argparse
import numpy as np
import subprocess as sb

from mpi4py import MPI

import model_builder as mdb
import project_tools as pjt

def simulation_script():
    sim_script ="#!/bin/bash\n"
    sim_script +="mdrun_sbm -s topol_4.5.tpr -table ../tables/table.xvg -tablep ../tables/tablep.xvg -tableb ../tables/table -maxh 23\n"
    sim_script +="""g_kuh_sbm -s Native.pdb -f traj.xtc -n native_contacts.ndx -o Q -noshortcut -abscut -cut 0.1
mv Q.out Q.dat
g_energy_sbm -f ener.edr -o energyterms -xvg none << EOF
Bond
Angle
Proper-Dih.
Gaussian
Potential
EOF"""
    return sim_script

def start_quenching_run(model,Tlist,pslist):

    nsteps = 600000
    name = model.name

    nvt = pjt.simulation.mdp.simulated_annealing(Tlist,pslist,str(nsteps))

    model.save_simulation_files(savetables=False)

    with open("nvt.mdp","w") as fout:
        # Molecular dynamics parameters
        fout.write(nvt)

    with open("prep.out","w") as fout:
        # Prepare run
        prep_step = 'grompp_sbm -n index.ndx -f nvt.mdp -c conf.gro -p topol.top -o topol_4.5.tpr '
        sb.call(prep_step.split(),stdout=fout,stderr=fout)

    with open("run.sh","w") as fout:
        fout.write(simulation_script())

    with open("sim.out","w") as fout:
        # Submit simulation and get jobID
        sbatch = "bash run.sh"
        sb.call(sbatch.split(),stdout=fout,stderr=fout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian estimation of 1D diffusion model.")
    parser.add_argument("--name",
                        type=str,
                        required=True,
                        help="Name of .ini file.")
    parser.add_argument("--rank_offset",
                        type=int,
                        required=True,
                        help="Increment the rank.")

    args = parser.parse_args()
    name = args.name
    rank_offset = args.rank_offset

    comm = MPI.COMM_WORLD   # MPI environment
    size = comm.Get_size()  # number of threads
    rank = comm.Get_rank()  # number of the current thread
    
    Tf = float(open("Tf","r").read().rstrip("\n"))

    Tlist = [2*Tf, 2*Tf, 0.1*Tf, 0.1*Tf]
    pslist = [0, 200, 10, 50]

    model,fitopts = mdb.inputs.load_model(name)

    simdir = "quench_%d" % (rank + rank_offset)
    if os.path.exists(simdir):
        print "skipping ", simdir
    else:
        print "starting ", simdir
        os.mkdir(simdir)
        os.chdir(simdir)
        # Run short quenching simulation
        start_quenching_run(model,Tlist,pslist)
        os.chdir("..")

