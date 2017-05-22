import os
import glob
import numpy as np

import mdtraj as md

if __name__ == "__main__":
    # get directories
    topname = "ref.pdb"
    tempdirs = glob.glob("T_*_1/traj.xtc") + glob.glob("T_*_2/traj.xtc") + glob.glob("T_*_3/traj.xtc")
    tempdirs = [ x.split("/traj.xtc")[0] for x in tempdirs ]
    organized_temps = util.get_organized_temps(temperature_dirs=tempdirs)
    T = organized_temps.keys()
    T.sort()
    topfile = tempdirs[0] + "/" + topname

    trajfiles = []
    for i in range(len(T)):
        trajfiles.append([ x + "/" + trajname for x in organized_temps[T[i]] ])

    for i in range(len(trajfiles)):
        traj = md.load(trajfiles[i], top=topfile)
        Rg = md.compute_rg(traj)
        np.save(os.path.dirname(trajfiles[i]) + "/Rg.npy", Rg)
