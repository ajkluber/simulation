import os
import numpy as np

import simulation.calc.observables as observables

if __name__ == "__main__":
    coordname = "Qtanh_0_05"
    trajfiles = [ x.rstrip("\n") for x in open("ticatrajs","r").readlines() ]
    dir = os.path.dirname(trajfiles[0])
    pairs = np.loadtxt("%s/native_contacts.ndx" % dir, skiprows=1, dtype=int) - 1
    n_native_pairs = pairs.shape[0]
    r0 = np.loadtxt("%s/pairwise_params" % dir, usecols=(4,))[1:2*n_native_pairs:2]
    top = "%s/Native.pdb" % dir
    widths = 0.05*np.ones(n_native_pairs, float)

    r0_cont = r0 + 0.1
    qtanh_obs = observables.TanhContacts(top, pairs, r0_cont, widths)
    #qtanh = observables.calculate_observable(trajfiles, qtanh_obs, saveas="Qtanh_0_05.dat")
    qtanh = [ np.loadtxt("%s/Qtanh_0_05.dat" % os.path.dirname(x)) for x in trajfiles ]

    n, bins= np.histogram(np.concatenate(qtanh),bins=30)
    bin_edges = np.array([ [bins[i], bins[i+1]] for i in range(len(bins) - 1 ) ])

    # Parameterize energy 
    nn_pair_type = np.loadtxt("%s/pairwise_params" % dir, usecols=(3,), dtype=int)[2*n_native_pairs::2]
    nn_pairs = np.loadtxt("%s/pairwise_params" % dir, usecols=(0,1), dtype=int)[2*n_native_pairs::2] - 1
    nn_eps = np.loadtxt("%s/model_params" % dir)[2*n_native_pairs::2]

    pair_params = []
    for line in open("%s/pairwise_params" % dir,"r"):
        if line.startswith("#"):
            continue
        else:
            pair_params.append(tuple([ float(x) for x in line.split()[4:] ]))
    nn_pair_params = pair_params[2*n_native_pairs::2]
    Enn_obs = observables.PairEnergySum(top, nn_pairs, nn_pair_type, nn_eps, nn_pair_params)

    if not os.path.exists("binned_Enn_vs_%s/Enn_vs_bin.dat" % coordname):
        print "binning Enn"
        Enn_bin_avg = observables.bin_observable(trajfiles, Enn_obs, qtanh, bin_edges)

        if not os.path.exists("binned_Enn_vs_%s" % coordname):
            os.mkdir("binned_Enn_vs_%s" % coordname)
        os.chdir("binned_Enn_vs_%s" % coordname)
        np.savetxt("Enn_vs_bin.dat",Enn_bin_avg)
        np.savetxt("bin_edges.dat",bin_edges)
        os.chdir("..")
    else:
        print "loading Enn"
        Enn_bin_avg = np.loadtxt("binned_Enn_vs_%s/Enn_vs_bin.dat" % coordname)

    if not os.path.exists("binned_Qtanhi_vs_%s/Qtanhi_vs_bin.dat" % coordname):
        print "binning qtanhi"
        qtanh_bin_avg = observables.bin_observable(trajfiles, qtanh_obs, qtanh, bin_edges)

        if not os.path.exists("binned_Qtanhi_vs_%s" % coordname):
            os.mkdir("binned_Qtanhi_vs_%s" % coordname)
        os.chdir("binned_Qtanhi_vs_%s" % coordname)
        np.savetxt("Qtanhi_vs_bin.dat",qtanh_bin_avg)
        np.savetxt("bin_edges.dat",bin_edges)
        os.chdir("..")
    else:
        print "loading qtanhi"
        qtanh_bin_avg = np.loadtxt("binned_Qtanhi_vs_%s/Qtanhi_vs_bin.dat" % coordname)

    # Calculate covariance
    print "calculating covariance"
    dEnndqi_bin_avg = observables.bin_observable_covariance(trajfiles, Enn_obs, qtanh_obs, Enn_bin_avg, qtanh_bin_avg, qtanh, bin_edges)

    if not os.path.exists("binned_covar_dEnndqi_vs_%s" % coordname):
        os.mkdir("binned_covar_dEnndqi_vs_%s" % coordname)
    os.chdir("binned_covar_dEnndqi_vs_%s" % coordname)
    np.savetxt("dEnndqi_vs_bin.dat",dEnndqi_bin_avg[:,0,:]) 
    np.savetxt("bin_edges.dat",bin_edges)
    os.chdir("..")
