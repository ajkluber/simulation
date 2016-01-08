import os
import numpy as np

import simulation.calc.observables as observables

def calc_Enn_for_directories(trajfiles, path_to_params=None):
    if path_to_params is None:
        path_to_params = os.path.dirname(trajfiles[0])

    nat_pairs = np.loadtxt("{}/native_contacts.ndx".format(path_to_params), skiprows=1, dtype=int) - 1
    n_native_pairs = nat_pairs.shape[0]

    top = "{}/Native.pdb".format(path_to_params)

    nn_pair_type = np.loadtxt("{}/pairwise_params".format(path_to_params), usecols=(3,), dtype=int)[2*n_native_pairs + 1::2]
    nn_pairs = np.loadtxt("{}/pairwise_params".format(path_to_params), usecols=(0,1), dtype=int)[2*n_native_pairs + 1::2] - 1
    nn_eps = np.loadtxt("{}/model_params".format(path_to_params))[2*n_native_pairs + 1::2]

    pair_params = []
    for line in open("{}/pairwise_params".format(path_to_params), "r"):
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
