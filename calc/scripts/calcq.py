import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mdtraj as md

import simulation.calc.observables as observables
import simulation.calc.pmfutil as pmfutil
import simulation.calc.util as util

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()

    name = args.name

    trajname = "traj.xtc"
    topname = "ref.pdb"
    coordfile = "Qtanh_0_05.npy"
    coordname = coordfile.split(".")[0]
    recalculate = True

    # get directories
    tempdirs = glob.glob("T_*_1") + glob.glob("T_*_2") + glob.glob("T_*_3")
    organized_temps = util.get_organized_temps(temperature_dirs=tempdirs)
    T = organized_temps.keys()
    T.sort()
    topfile = tempdirs[0] + "/" + topname

    trajfiles = []
    for i in range(len(T)):
        trajfiles.append([ x + "/" + trajname for x in organized_temps[T[i]] ])

    # parameterize native contacts function
    if os.path.exists(name + ".contacts"):
        pairs = np.loadtxt(name + ".contacts", dtype=int) - 1
    elif os.path.exists("../../" + name + ".contacts"):
        pairs = np.loadtxt("../../" + name + ".contacts", dtype=int) - 1

    if pairs.shape[1] == 4:
        pairs = pairs[:,(1,3)]
    n_native_pairs = len(pairs)
    
    r0 = md.compute_distances(md.load(topfile), pairs)
    r0_cont = r0 + 0.1
    widths = 0.05*np.ones(n_native_pairs)
    qtanhsum_obs = observables.TanhContactSum(topfile, pairs, r0_cont, widths, periodic=True)

    for i in range(len(T)):
        print T[i]
        qtanh_files = [ "{}/{}".format(os.path.dirname(x),coordfile) for x in trajfiles[i] ]
        qtanh_files_exist = [ os.path.exists(x) for x in qtanh_files ]
        # this should work except I am experiencing a strange NaN error.
        if np.all(qtanh_files_exist) and not recalculate:  
            qtanh = [ np.load(x) for x in qtanh_files ]
        else:
            qtanh = []
            for n in range(len(trajfiles[i])):
                traj = trajfiles[i][n]
                qtanh_temp = np.array(qtanhsum_obs.map(md.load(traj, top=topfile)))
                np.save(os.path.dirname(traj) + "/" + coordfile, qtanh_temp)
                qtanh.append(qtanh_temp)

        # calculate free energy profile and state definitions
        mid_bin, Fdata = pmfutil.pmf1D(np.concatenate(qtanh), bins=40)
        #plt.plot(mid_bin, Fdata, label=str(T[i]))

        if not os.path.exists(coordname + "_profile"):
            os.mkdir(coordname + "_profile")
        os.chdir(coordname + "_profile")
        np.savetxt("T_{}_mid_bin.dat".format(T[i]), mid_bin)
        np.savetxt("T_{}_F.dat".format(T[i]), Fdata)

        xinterp, F = pmfutil.interpolate_profile(mid_bin, Fdata)
        minidx, maxidx = pmfutil.extrema_from_profile(xinterp, F)

        F_fit = F(xinterp)
        with open("T_{}_stab.dat".format(T[i]), "w") as fout:
            fout.write(str(F_fit[minidx[1]] - F_fit[minidx[0]]))
         
        os.chdir("..")

