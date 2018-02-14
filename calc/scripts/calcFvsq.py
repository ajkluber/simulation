import os
import numpy as np

import simulation.calc.pmfutil as pmfutil

if __name__ == "__main__":
    coordfile = "Qtanh_0_05.npy"
    coordname = coordfile.split(".")[0]

    with open("{}_profile/T_used.dat".format(coordname), "r") as fin:
        T_used = float(fin.read())
        
    qtrajs = [ np.load("T_{:.2f}_{}/{}".format(T_used, x, coordfile)) for x in [1,2,3] ]

    # calculate free energy profile and state definitions
    mid_bin, Fdata = pmfutil.pmf1D(np.concatenate(qtrajs), bins=40)

    if not os.path.exists(coordname + "_profile"):
        os.mkdir(coordname + "_profile")

    os.chdir(coordname + "_profile")
    np.savetxt("T_{}_mid_bin.dat".format(T_used), mid_bin)
    np.savetxt("T_{}_F.dat".format(T_used), Fdata)

    xinterp, F = pmfutil.interpolate_profile(mid_bin, Fdata)
    minidx, maxidx = pmfutil.extrema_from_profile(xinterp, F)

    F_fit = F(xinterp)
    with open("T_{}_stab.dat".format(T_used), "w") as fout:
        fout.write(str(F_fit[minidx[1]] - F_fit[minidx[0]]))
     
    os.chdir("..")
