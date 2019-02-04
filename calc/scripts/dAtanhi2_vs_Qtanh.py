import os
import numpy as np

import simulation.calc.observables as observables
import simulation.util as util

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", 
            type=str, 
            help="Name")

    parser.add_argument("n_native_pairs", 
            type=float, 
            help="Numer of native contacts")

    args = parser.parse_args()
    name = args.name
    n_native_pairs = float(args.n_native_pairs)

    T_used = util.get_T_used()
    with open("Qtanh_0_05_profile/T_used.dat", "r") as fin:
        T_used = float(fin.read())
    
    trajfiles = [ "T_{:.2f}_{}/traj.xtc".format(T_used, x) for x in [1,2,3]]
    dir = os.path.dirname(trajfiles[0])
    top = "{}/ref.pdb".format(dir)

    pairs = np.loadtxt("{}_pairwise_params".format(name), usecols=(0,1), dtype=int)
    r0 = np.loadtxt("{}_pairwise_params".format(name), usecols=(4,))
    r0_cont = r0 + 0.1

    nn_pairs = np.loadtxt("%s/pairwise_params" % dir, usecols=(0,1))[2*n_native_pairs + 1::2] - 1
    nn_r0 = np.loadtxt("%s/pairwise_params" % dir, usecols=(4,))[2*n_native_pairs + 1::2]
    nn_r0_cont = nn_r0 + 0.1
    widths = 0.05


    qtanh = [ np.load("%s/Qtanh_0_05.npy" % x.split("/")[0]) for x in trajfiles ]


    Atanhi_obs = observables.TanhContacts(top, nn_pairs, nn_r0_cont, widths)
    if os.path.exists("binned_Ai_vs_Qtanh_0_05/Ai_vs_bin.dat"):
        os.chdir("binned_Ai_vs_Qtanh_0_05")
        mid_bin = np.loadtxt("mid_bin.dat")
        Atanhi_bin_avg = np.loadtxt("Ai_vs_bin.dat")
        os.chdir("..")
        
        bins = np.zeros(mid_bin.shape[0] + 1, float)
        bins[1:-1] = 0.5*(mid_bin[1:] + mid_bin[:-1])
        bins[0] = mid_bin[0] - (bins[1] - mid_bin[0])
        bins[-1] = mid_bin[-1] + (mid_bin[-1] - bins[-2])
    else:
        n, bins = np.histogram(np.concatenate(qtanh),bins=40)
        mid_bin = 0.5*(bins[1:] + bins[:-1])

        Atanhi_bin_avg = observables.bin_observable(trajfiles, Atanhi_obs, qtanh, bin_edges)

        if not os.path.exists("binned_Ai_vs_Qtanh_0_05"):
            os.mkdir("binned_Ai_vs_Qtanh_0_05")
        os.chdir("binned_Ai_vs_Qtanh_0_05")
        np.savetxt("mid_bin.dat",mid_bin)
        np.savetxt("Ai_vs_bin.dat",Atanhi_bin_avg)
        os.chdir("..")

    bin_edges = np.array([ [bins[i], bins[i+1]] for i in range(len(bins) - 1 ) ])

    # Calculate variance in Ai
    dAtanhi2_bin_avg = observables.bin_observable_variance(trajfiles, Atanhi_obs, Atanhi_bin_avg, qtanh, bin_edges)

    if not os.path.exists("binned_dAi2_vs_Qtanh_0_05"):
        os.mkdir("binned_dAi2_vs_Qtanh_0_05")
    os.chdir("binned_dAi2_vs_Qtanh_0_05")
    np.savetxt("mid_bin.dat",mid_bin)
    np.savetxt("dAi2_vs_bin.dat",dAtanhi2_bin_avg)
    os.chdir("..")
