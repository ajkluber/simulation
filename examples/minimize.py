import shutil
import os
import glob
import subprocess as sb
import numpy as np

import mdtraj 

import simulation.mdp
import simulation.slurm

import model_builder as mdb

def minimization_script(frame_idx):
    """Takes a starting structure and energy minimizes it."""
    script = \
"""#!/bin/bash
# Run energy minimization
grompp_sbm -n index.ndx -f em.mdp -c conf.gro -p topol.top -o topol_4.5.tpr &> grompp.log
mdrun_sbm -s topol_4.5.tpr -table tables/table.xvg -tablep tables/tablep.xvg -tableb tables/table &> mdrun.log

# Get the last frame of the minimization
g_energy_sbm -f ener.edr -o Etot -xvg none &> energy.log << EOF
Potential
EOF
tail -n 1 Etot.xvg | awk '{ print $(NF) }' > Etot_%d.dat
fstep=`grep "Low-Memory BFGS Minimizer converged" md.log | awk '{ print $(NF-1) }'`
trjconv_sbm -f traj.trr -s topol_4.5.tpr -n index.ndx -o frame_%d.xtc -dump ${fstep} &> trjconv.log << EOF
System
EOF

# Cleanup afterwards
rm conf.gro mdout.mdp topol_4.5.tpr traj.trr md.log ener.edr confout.gro Etot.xvg
rm grompp.log mdrun.log energy.log trjconv.log
""" % (frame_idx, frame_idx)
    return script

if __name__ == "__main__":
    import time
    starttime = time.time()

    if not os.path.exists("inherent_structures"):
        os.mkdir("inherent_structures")
    os.chdir("inherent_structures")

    # Run parameters
    mdp = simulation.mdp.energy_minimization()
    with open("em.mdp", "w") as fout:
        fout.write(mdp)

    # Load model
    cwd = os.getcwd()
    model_dir = "/home/ajk8/scratch/6-10-15_nonnative/1E0G/random_b2_0.01/replica_1"
    os.chdir(model_dir)
    model, fitopts = mdb.inputs.load_model("1E0G")
    os.chdir(cwd)

    # Save model files
    model.save_simulation_files(savetables=False)
    if not os.path.exists("tables"):
        os.mkdir("tables")
    os.chdir("tables")
    np.savetxt("table.xvg", model.tablep, fmt="%16.15e", delimiter=" ")
    np.savetxt("tablep.xvg", model.tablep, fmt="%16.15e", delimiter=" ")
    for i in range(model.n_tables):
        np.savetxt(model.tablenames[i], model.tables[i], fmt="%16.15e", delimiter=" ")
    os.chdir("..")

    preptime = time.time()
    print "Preparation took: %.4f min" % ((preptime - starttime)/60.)
    
    # How to determine length of trajectory? Assume frame_idxs given
    frame_idxs = range(0, 600000, 210)
    np.savetxt("frame_idxs.dat", np.array(frame_idxs), fmt="%d")
    #frame_idxs = range(1200, 1200 + 600, 10)
    n_frames = len(frame_idxs)
    # Loop over trajectory frames
    for i in range(len(frame_idxs)):
        idx = frame_idxs[i]
        print "minimizing frame: %d" % idx

        # slice frame from trajectory
        frm = mdtraj.load_frame("../traj.xtc", idx, top="../Native.pdb")
        frm.save_gro("conf.gro")

        # perform energy minimization using gromacs
        script = minimization_script(idx)
        with open("minimize.bash", "w") as fout:
            fout.write(script)
        sb.call("bash minimize.bash".split())

    calctime = time.time()
    calcmin = (calctime - starttime)/60.
    print "Calculation took: %.4f min" % (calcmin)
    print "Avg. calc rate for %d frames: %.4f sec/frame" %  (n_frames, (60.*calcmin/n_frames))
    os.chdir("..")
