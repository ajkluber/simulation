import os
import glob
import shutil
import argparse
import numpy as np
import subprocess as sb

from mpi4py import MPI

import mdtraj 

import simulation.mdp
import simulation.slurm

import model_builder as mdb

def minimization_script():
    """Takes a starting structure and energy minimizes it."""
    script = \
"""#!/bin/bash
# run energy minimization
grompp_sbm -n ../index.ndx -f ../em.mdp -c conf.gro -p ../topol.top -o topol_4.5.tpr &> grompp.log
mdrun_sbm -s topol_4.5.tpr -table ../tables/table.xvg -tablep ../tables/tablep.xvg -tableb ../tables/table &> mdrun.log

# get final energy 
g_energy_sbm -f ener.edr -o Energy -xvg none &> energy.log << EOF
Potential
EOF
tail -n 1 Energy.xvg | awk '{ print $(NF) }' >> Etot.dat

# get final structure 
fstep=`grep "Low-Memory BFGS Minimizer converged" md.log | awk '{ print $(NF-1) }'`
trjconv_sbm -f traj.trr -s topol_4.5.tpr -n ../index.ndx -o temp_frame.xtc -dump ${fstep} &> trjconv.log << EOF
System
EOF

# concatenate to trajectory
if [ ! -e all_frames.xtc ]; then
    mv temp_frame.xtc all_frames.xtc
else
    trjcat_sbm -f all_frames.xtc temp_frame.xtc -o all_frames.xtc -n ../index.ndx -nosort -cat &> trjcat.log << EOF
System
EOF
    rm temp_frame.xtc
fi

# cleanup
rm conf.gro mdout.mdp topol_4.5.tpr traj.trr md.log ener.edr confout.gro Energy.xvg
"""
    return script

def prep_minimization(model_dir, name):
    """Save model files if needed"""

    # Run parameters
    mdp = simulation.mdp.energy_minimization()
    with open("em.mdp", "w") as fout:
        fout.write(mdp)

    # Load model
    cwd = os.getcwd()
    os.chdir(model_dir)
    model, fitopts = mdb.inputs.load_model(name)
    os.chdir(cwd)

    # Save model files
    model.save_simulation_files(savetables=False)
    if not os.path.exists("tables"):
        os.mkdir("tables")
    os.chdir("tables")
    if not os.path.exists("table.xvg"):
        np.savetxt("table.xvg", model.tablep, fmt="%16.15e", delimiter=" ")
    if not os.path.exists("tablep.xvg"):
        np.savetxt("tablep.xvg", model.tablep, fmt="%16.15e", delimiter=" ")
    for i in range(model.n_tables):
        if not os.path.exists(model.tablenames[i]):
            np.savetxt(model.tablenames[i], model.tables[i], fmt="%16.15e", delimiter=" ")
    os.chdir("..")

def run_minimization(frame_idxs, trajfile="../../traj.xtc", top="../../Native.pdb"):
    """Perform energy minimization on each frame"""

    comm = MPI.COMM_WORLD   
    size = comm.Get_size()  
    rank = comm.Get_rank()

    # split frames equally among processors
    frames_per_proc = len(frame_idxs)/size
    
    if rank == (size - 1):
        frame_idxs_thread = np.array(frame_idxs[rank*frames_per_proc:])
    else:
        frame_idxs_thread = np.array(frame_idxs[rank*frames_per_proc:(rank + 1)*frames_per_proc])

    if not os.path.exists("rank_{}".format(rank)):
        os.mkdir("rank_{}".format(rank))
    os.chdir("rank_{}".format(rank))

    np.savetxt("frame_idxs.dat", frame_idxs_thread, fmt="%d")

    # Loop over trajectory frames
    for i in xrange(len(frame_idxs_thread)):
        idx = frame_idxs_thread[i]
        # slice frame from trajectory
        print "minimizing frame: %d" % idx
        frm = mdtraj.load_frame(trajfile, idx, top=top)
        frm.save_gro("conf.gro")

        # perform energy minimization using gromacs
        script = minimization_script()
        with open("minimize.bash", "w") as fout:
            fout.write(script)
        cmd = "bash minimize.bash"
        sb.call(cmd.split())

    os.chdir("..")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Energy minimization for inherent structure analysis.")
    parser.add_argument("--name",
                        type=str,
                        required=True,
                        help="Name of .ini file.")

    parser.add_argument("--path_to_ini",
                        type=str,
                        required=True,
                        help="Path to .ini file.")

    parser.add_argument("--skip",
                        type=int,
                        default=10,
                        help="Number of frames to skip. Subsample.")

    parser.add_argument("--n_frames",
                        type=int,
                        default=int(6E5),
                        help="Number of frames in trajectory.")
    
    args = parser.parse_args()
    name = args.name
    model_dir = args.path_to_ini
    n_frames = args.n_frames
    skip = args.skip

    # Performance on one processor is roughly 11sec/frame. 
    # So 1proc can do about 2500 frames over 8hours.
    # Adjust the number of processors (size) and subsample (skip)
    # accordingingly

    #name = "1E0G"
    #model_dir = "/home/ajk8/scratch/6-10-15_nonnative/1E0G/random_b2_0.01/replica_1"
    #model_dir = "/home/ajk8/scratch/6-10-15_nonnative/1E0G/random_b2_1.00/replica_1"
    #n_frames = int(6E5)
    #skip = 10

    frame_idxs = range(0, n_frames, skip)

    comm = MPI.COMM_WORLD   
    size = comm.Get_size()  
    rank = comm.Get_rank()

    if rank == 0:
        if not os.path.exists("inherent_structures"):
            os.mkdir("inherent_structures")
    comm.Barrier()

    os.chdir("inherent_structures")
    prep_minimization(model_dir, name)
    run_minimization(frame_idxs)
    os.chdir("..")
