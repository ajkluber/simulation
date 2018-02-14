import os
import glob
import numpy as np

import simulation.calc.pmfutil as pmfutil
import simulation.calc.transits as transits
import simulation.calc.util as util

if __name__ == "__main__":
    coordfile = "Qtanh_0_05.npy"
    coordname = coordfile.split(".")[0]
    #coordfile = "qtanh.npy"

    # get directories
    tempdirs = glob.glob("T_*_1") + glob.glob("T_*_2") + glob.glob("T_*_3")
    organized_temps = util.get_organized_temps(temperature_dirs=tempdirs)
    T = organized_temps.keys()
    T.sort()

    # determine which temperature is closest to the folding temperature
    os.chdir(coordname + "_profile")
    if not (os.path.exists("T_used.dat") and os.path.exists("minima.dat")):
        dF_min_idx = np.argmin([ (float(open("T_{}_stab.dat".format(x)).read()))**2 for x in T ])
        T_used = T[dF_min_idx] 
        mid_bin = np.loadtxt("T_{}_mid_bin.dat".format(T_used))
        Fdata = np.loadtxt("T_{}_F.dat".format(T_used))
        xinterp, F = pmfutil.interpolate_profile(mid_bin, Fdata)
        minidx, maxidx = pmfutil.extrema_from_profile(xinterp, F)
        np.savetxt("minima.dat", xinterp[minidx])
        with open("T_used.dat", "w") as fout:
            fout.write(str(T_used))
        U = xinterp[minidx].min()
        N = xinterp[minidx].max()
    else:
        with open("T_used.dat", "r") as fin:
            T_used = float(fin.read())
        minima = np.loadtxt("minima.dat")
        U = minima.min()
        N = minima.max()

    os.chdir("..")

    dirs = organized_temps[T_used]

    # calculate transit time between minima for each trajectory.
    all_dwellsU = []
    all_dwellsN = []
    all_transitsUN = []
    all_transitsNU = []
    for i in range(len(dirs)):
        print dirs[i]
        os.chdir(dirs[i])
        if coordfile.endswith(".npy"):
            x = np.load(coordfile)
        else:
            x = np.loadtxt(coordfile)
        print x.shape
        dtraj = np.zeros(x.shape[0], int)
        dtraj[x <= U] = 0
        dtraj[x >= N] = 2
        dtraj[(x > U) & (x < N)] = 1

        # calculate transits
        dwellsU, dwellsN, transitsUN, transitsNU = transits.partition_dtraj(dtraj, 0, 2)

        os.chdir("..")

        all_dwellsU.append(dwellsU)
        all_dwellsN.append(dwellsN)
        all_transitsUN.append(transitsUN)
        all_transitsNU.append(transitsNU)

    folding_time = np.concatenate([ all_dwellsU[i][:,1] + all_transitsUN[i][:,1] for i in range(len(all_dwellsU)) ])
    unfolding_time = np.concatenate([ all_dwellsN[i][:,1] + all_transitsNU[i][:,1] for i in range(len(all_dwellsN)) ])

    forward_transit_time = np.concatenate([ all_transitsUN[i][:,1] for i in range(len(all_transitsUN)) ])
    backward_transit_time = np.concatenate([ all_transitsNU[i][:,1] for i in range(len(all_transitsNU)) ])

    # Save transit times
    if not os.path.exists("%s_transit_time" % coordname): 
        os.mkdir("%s_transit_time" % coordname)
    os.chdir("%s_transit_time" % coordname)
    with open("forward_mean","w") as fout:
        fout.write("%.2f" % np.mean(forward_transit_time))
    with open("backward_mean","w") as fout:
        fout.write("%.2f" % np.mean(backward_transit_time))
    np.savetxt("forward_transit_times.dat", forward_transit_time, fmt="%5d")
    np.savetxt("backward_transit_times.dat", backward_transit_time, fmt="%5d")
    os.chdir("..")

    # Save folding times
    if not os.path.exists("%s_folding_time" % coordname): 
        os.mkdir("%s_folding_time" % coordname)
    os.chdir("%s_folding_time" % coordname)
    with open("folding_mean","w") as fout:
        fout.write("%.2f" % np.mean(folding_time))
    with open("unfolding_mean","w") as fout:
        fout.write("%.2f" % np.mean(unfolding_time))
    np.savetxt("folding_times.dat", folding_time, fmt="%5d")
    np.savetxt("unfolding_times.dat", unfolding_time, fmt="%5d")
    os.chdir("..")
