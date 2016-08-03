import os
import numpy as np        
import matplotlib.pyplot as plt

import plotter.pmfutil as pmfutil

if __name__ == "__main__":
    coordfile = "Qtanh_0_05.dat"
    Tdirs = [ x.rstrip("\n") for x in open("ticatemps","r").readlines() ]


    tempdirs = glob.glob("T_*_1") + glob.glob("T_*_2") + glob.glob("T_*_3")

    T = Tdirs[0].split("_")[-2]
    coordname = coordfile.split(".")[0]

    coordvst = np.concatenate([np.loadtxt("%s/%s" % (x,coordfile)) for x in Tdirs ])
    if not os.path.exists("%s_profile" % coordname):
        os.mkdir("%s_profile" % coordname)
    os.chdir("%s_profile" % coordname)
    mid_bin, Fdata = pmfutil.pmf1D(coordvst,bins=40)
    np.savetxt("mid_bin.dat",mid_bin)
    np.savetxt("F.dat",Fdata)
    xinterp, F = pmfutil.interpolate_profile(mid_bin,Fdata)
    minidx, maxidx = pmfutil.extrema_from_profile(xinterp,F)
    min_bounds, max_bounds = pmfutil.state_bounds_from_profile(xinterp,F)
    min_labels, max_labels = pmfutil.assign_state_labels(min_bounds,max_bounds)
    pmfutil.save_state_bounds(T, min_bounds, max_bounds, min_labels, max_labels)
    with open("T.txt", "w") as fout:
        fout.write("%s" % T)
    os.chdir("..")

