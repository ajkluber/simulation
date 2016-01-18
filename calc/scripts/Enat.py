import os
import numpy as np

import simulation.calc.observables as observables

def calc_Enat_for_directories(trajfiles, path_to_params=None):
    if path_to_params is None:
        path_to_params = os.path.dirname(trajfiles[0])

    nat_pairs = np.loadtxt("{}/native_contacts.ndx".format(path_to_params), skiprows=1, dtype=int) - 1
    n_native_pairs = nat_pairs.shape[0]

    top = "{}/Native.pdb".format(path_to_params)

    # Parameterize native energy 
    nat_pair_type = np.loadtxt("%s/pairwise_params" % dir, usecols=(3,), dtype=int)[1:2*n_native_pairs:2]
    nat_pairs = np.loadtxt("%s/pairwise_params" % dir, usecols=(0,1), dtype=int)[1:2*n_native_pairs:2] - 1
    nat_eps = np.loadtxt("%s/model_params" % dir)[1:2*n_native_pairs:2]

    pair_params = []
    for line in open("{}/pairwise_params".format(path_to_params), "r"):
        if line.startswith("#"):
            continue
        else:
            pair_params.append(tuple([ float(x) for x in line.split()[4:] ]))
    nat_pair_params = pair_params[1:2*n_native_pairs:2]

    Enat_obs = observables.PairEnergySum(top, nat_pairs, nat_pair_type, nat_eps, nat_pair_params)
    Enat = observables.calculate_observable(trajfiles, Enat_obs, saveas="Enative.dat")

if __name__ == "__main__":
    trajfiles = [ x.rstrip("\n") for x in open("ticatrajs","r").readlines() ]

    calc_Enat_for_directories(trajfiles)
