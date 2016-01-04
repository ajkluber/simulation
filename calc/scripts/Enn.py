import os
import numpy as np

import simulation.calc.observables as observables

def calc_Enn_for_directories(trajfiles):
    dir = os.path.dirname(trajfiles[0])
    nat_pairs = np.loadtxt("%s/native_contacts.ndx" % dir, skiprows=1, dtype=int) - 1
    n_native_pairs = nat_pairs.shape[0]

    top = "%s/Native.pdb" % dir

    nn_pair_type = np.loadtxt("%s/pairwise_params" % dir, usecols=(3,), dtype=int)[2*n_native_pairs + 1::2]
    nn_pairs = np.loadtxt("%s/pairwise_params" % dir, usecols=(0,1), dtype=int)[2*n_native_pairs + 1::2] - 1
    nn_eps = np.loadtxt("%s/model_params" % dir)[2*n_native_pairs + 1::2]

    pair_params = []
    for line in open("%s/pairwise_params" % dir,"r"):
        if line.startswith("#"):
            continue
        else:
            pair_params.append(tuple([ float(x) for x in line.split()[4:] ]))
    nn_pair_params = pair_params[2*n_native_pairs + 1::2]

    Enn_obs = observables.PairEnergySum(top, nn_pairs, nn_pair_type, nn_eps, nn_pair_params)
    Enn = observables.calculate_observable(trajfiles, Enn_obs, saveas="Enonnative.dat")

if __name__ == "__main__":
    trajfiles = [ x.rstrip("\n") for x in open("ticatrajs","r").readlines() ]

    calc_Enn_for_directories(trajfiles)
