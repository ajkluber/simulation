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

    if all([ (os.path.exists("%s/Atanh_0_05.dat" % x.split("/")[0]) & os.path.exists("%s/Qtanh_0_05.dat" % x.split("/")[0])) \
                for x in trajfiles ]):

        qtanh = [ np.loadtxt("%s/Qtanh_0_05.dat" % x.split("/")[0]) for x in trajfiles ]
        Atanh = [ np.loadtxt("%s/Atanh_0_05.dat" % x.split("/")[0]) for x in trajfiles ]
    else:
        qtanhsum_obs = observables.TanhContactSum(top, pairs, r0_cont, widths)
        qtanh = observables.calculate_observable(trajfiles, qtanhsum_obs, saveas="Qtanh_0_05.dat")

        Atanhsum_obs = observables.TanhContactSum(top, nn_pairs, nn_r0_cont, widths)
        Atanh = observables.calculate_observable(trajfiles, Atanhsum_obs, saveas="Atanh_0_05.dat")

    q = np.concatenate(qtanh)
    A = np.concatenate(Atanh)
    n,bin_edges = np.histogram(q,bins=40)
    mid_bin = 0.5*(bin_edges[1:] + bin_edges[:-1])

    A_bin_avg = np.zeros(len(bin_edges) - 1,float)
    dA2_bin_avg = np.zeros(len(bin_edges) - 1,float)
    for i in range(len(bin_edges) - 1):
        frames_in_this_bin = ((q > bin_edges[i]) & (q <= bin_edges[i+1]))
        if any(frames_in_this_bin):
            A_bin_avg[i] = np.mean(A[frames_in_this_bin])
            dA2_bin_avg[i] = np.mean((A[frames_in_this_bin] - A_bin_avg[i])**2)

    if not os.path.exists("binned_A_vs_Qtanh_0_05"):
        os.mkdir("binned_A_vs_Qtanh_0_05")
    os.chdir("binned_A_vs_Qtanh_0_05")
    np.savetxt("mid_bin.dat",mid_bin)
    np.savetxt("A_vs_bin.dat",A_bin_avg)
    os.chdir("..")

    if not os.path.exists("binned_dA2_vs_Qtanh_0_05"):
        os.mkdir("binned_dA2_vs_Qtanh_0_05")
    os.chdir("binned_dA2_vs_Qtanh_0_05")
    np.savetxt("mid_bin.dat",mid_bin)
    np.savetxt("dA2_vs_bin.dat",dA2_bin_avg)
    os.chdir("..")
