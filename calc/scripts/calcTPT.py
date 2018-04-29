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
        print len(xinterp), minidx, xinterp[minidx]

        # don't count minima at the ends of data 
        minidx = np.array([ idx for idx in minidx if (10 < idx < (len(xinterp) - 10)) ])
        maxidx = np.array([ idx for idx in maxidx if (10 < idx < (len(xinterp) - 10)) ])
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

    has_events_U = np.array([ len(all_dwellsU[i]) > 0 for i in range(len(all_dwellsU)) ])
    has_events_N = np.array([ len(all_dwellsN[i]) > 0 for i in range(len(all_dwellsN)) ])

    if np.any(has_events_U):
        folding_time = np.concatenate([ all_dwellsU[i][:,1] + all_transitsUN[i][:,1] for i in range(len(all_dwellsU)) if has_events_U[i]])
        forward_transit_time = np.concatenate([ all_transitsUN[i][:,1] for i in range(len(all_transitsUN)) if has_events_U[i]])
    if np.any(has_events_N):
        unfolding_time = np.concatenate([ all_dwellsN[i][:,1] + all_transitsNU[i][:,1] for i in range(len(all_dwellsN)) if has_events_N[i]])
        backward_transit_time = np.concatenate([ all_transitsNU[i][:,1] for i in range(len(all_transitsNU)) if has_events_N[i]])

    tp_dir = "{}_transit_time".format(coordname)
    tf_dir = "{}_folding_time".format(coordname)
    if not os.path.exists(tp_dir): 
        os.mkdir(tp_dir)
    if not os.path.exists(tf_dir): 
        os.mkdir(tf_dir)

    # If there are folding/unfolding events save transit and folding times
    if np.any(has_events_U):
        with open(tp_dir + "/forward_mean","w") as fout:
            fout.write("%.2f" % np.mean(forward_transit_time))
        np.savetxt(tp_dir + "/forward_transit_times.dat", forward_transit_time, fmt="%5d")

        with open(tf_dir + "/folding_mean","w") as fout:
            fout.write("%.2f" % np.mean(folding_time))
        np.savetxt(tf_dir + "/folding_times.dat", folding_time, fmt="%5d")


    if np.any(has_events_N):
        with open(tp_dir + "/backward_mean","w") as fout:
            fout.write("%.2f" % np.mean(backward_transit_time))
        np.savetxt(tp_dir + "/backward_transit_times.dat", backward_transit_time, fmt="%5d")

        with open(tf_dir + "/unfolding_mean","w") as fout:
            fout.write("%.2f" % np.mean(unfolding_time))
        np.savetxt(tf_dir + "/unfolding_times.dat", unfolding_time, fmt="%5d")
