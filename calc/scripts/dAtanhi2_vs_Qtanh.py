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

    nn_pairs = np.loadtxt("%s/pairwise_params" % dir, usecols=(0,1))[2*n_native_pairs + 1::2] - 1
    nn_r0 = np.loadtxt("%s/pairwise_params" % dir, usecols=(4,))[2*n_native_pairs + 1::2]
    nn_r0_cont = nn_r0 + 0.1
    widths = 0.05

    top = "%s/Native.pdb" % dir

    if all([ os.path.exists("%s/Qtanh_0_05.dat" % x.split("/")[0]) for x in trajfiles ]):
        qtanh = [ np.loadtxt("%s/Qtanh_0_05.dat" % x.split("/")[0]) for x in trajfiles ]
    else:
        qtanhsum_obs = observables.TanhContactSum(top, pairs, r0_cont, widths)
        qtanh = observables.calculate_observable(trajfiles, qtanhsum_obs, saveas="Qtanh_0_05.dat")

    n, bins= np.histogram(np.concatenate(qtanh),bins=40)
    mid_bin = 0.5*(bins[1:] + bins[:-1])
    bin_edges = np.array([ [bins[i], bins[i+1]] for i in range(len(bins) - 1 ) ])

    Atanhi_obs = observables.TanhContacts(top, nn_pairs, nn_r0_cont, widths)
    Atanhi_bin_avg = observables.bin_observable(trajfiles, Atanhi_obs, qtanh, bin_edges)

    if not os.path.exists("binned_Ai_vs_Qtanh_0_05"):
        os.mkdir("binned_Ai_vs_Qtanh_0_05")
    os.chdir("binned_Ai_vs_Qtanh_0_05")
    np.savetxt("mid_bin.dat",mid_bin)
    np.savetxt("Ai_vs_bin.dat",Atanhi_bin_avg)
    os.chdir("..")

    # Calculate variance in Ai
    dAtanhi2_bin_avg = observables.bin_observable_covariance(trajfiles, Atanhi_obs, Atanhi_obs, Atanhi_bin_avg, Atanhi_bin_avg, qtanh, bin_edges)

    if not os.path.exists("binned_dAi2_vs_Qtanh_0_05"):
        os.mkdir("binned_dAi2_vs_Qtanh_0_05")
    os.chdir("binned_dAi2_vs_Qtanh_0_05")
    np.savetxt("mid_bin.dat",mid_bin)
    np.savetxt("dAi2_vs_bin.dat",dAtanhi2_bin_avg)
    os.chdir("..")
