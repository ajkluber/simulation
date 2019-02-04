import os
import glob
import numpy as np

import mdtraj as md

import simulation.calc.util as util

if __name__ == "__main__":
    # get directories
    topname = "ref.pdb"
    trajname = "traj.xtc"

    with open("Qtanh_0_05_profile/T_used.dat", "r") as fin:
        T_used = float(fin.read())
    trajfiles = [ "T_{:.2f}_{}/traj.xtc".format(T_used, x) for x in [1,2,3]]
    tempdirs = [ x.split("/")[0] for x in trajfiles ]
    topfile = tempdirs[0] + "/" + topname

    for i in range(len(trajfiles)):
        traj = md.load(trajfiles[i], top=topfile)
        Rg = md.compute_rg(traj)
        np.save(tempdirs[i] + "/Rg.npy", Rg)
