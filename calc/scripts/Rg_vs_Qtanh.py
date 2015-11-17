import os
import numpy as np

import simulation.calc.observables as observables

if __name__ == "__main__":
    trajfiles = [ x.rstrip("\n") for x in open("ticatrajs","r").readlines() ]
    dir = os.path.dirname(trajfiles[0])
    pairs = np.loadtxt("%s/native_contacts.ndx" % dir, skiprows=1, dtype=int) - 1
    n_native_pairs = pairs.shape[0]
    r0 = np.loadtxt("%s/pairwise_params" % dir, usecols=(4,))[1:2*n_native_pairs:2]
    r0_cont = r0 + 0.1

    widths = 0.05

    top = "%s/Native.pdb" % dir

    if all([ (os.path.exists("%s/Rg.dat" % x.split("/")[0]) & os.path.exists("%s/Qtanh_0_05.dat" % x.split("/")[0])) \
                for x in trajfiles ]):
        qtanh = [ np.loadtxt("%s/Qtanh_0_05.dat" % x.split("/")[0]) for x in trajfiles ]
        rg = [ np.loadtxt("%s/Rg.dat" % x.split("/")[0]) for x in trajfiles ]
    else:
        raise IOError("Rg.dat does not exist!")
        #qtanhsum_obs = observables.TanhContactSum(top, pairs, r0_cont, widths)
        #qtanh = observables.calculate_observable(trajfiles, qtanhsum_obs, saveas="Qtanh_0_05.dat")

    q = np.concatenate(qtanh)
    Rg = np.concatenate(rg)
    n,bin_edges = np.histogram(q,bins=40)
    mid_bin = 0.5*(bin_edges[1:] + bin_edges[:-1])

    Rg_bin_avg = np.zeros(len(bin_edges) - 1,float)
    dRg2_bin_avg = np.zeros(len(bin_edges) - 1,float)
    for i in range(len(bin_edges) - 1):
        frames_in_this_bin = ((q > bin_edges[i]) & (q <= bin_edges[i+1]))
        if any(frames_in_this_bin):
            Rg_bin_avg[i] = np.mean(Rg[frames_in_this_bin])
            dRg2_bin_avg[i] = np.mean((Rg[frames_in_this_bin] - Rg_bin_avg[i])**2)

    if not os.path.exists("binned_Rg_vs_Qtanh_0_05"):
        os.mkdir("binned_Rg_vs_Qtanh_0_05")
    os.chdir("binned_Rg_vs_Qtanh_0_05")
    np.savetxt("mid_bin.dat",mid_bin)
    np.savetxt("Rg_vs_bin.dat",Rg_bin_avg)
    os.chdir("..")

    if not os.path.exists("binned_dRg2_vs_Qtanh_0_05"):
        os.mkdir("binned_dRg2_vs_Qtanh_0_05")
    os.chdir("binned_dRg2_vs_Qtanh_0_05")
    np.savetxt("mid_bin.dat",mid_bin)
    np.savetxt("dRg2_vs_bin.dat",dRg2_bin_avg)
    os.chdir("..")
