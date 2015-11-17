import os
import numpy as np

import simulation.calc.observables as observables

if __name__ == "__main__":
    trajfiles = [ x.rstrip("\n") for x in open("ticatrajs","r").readlines() ]
    dir = os.path.dirname(trajfiles[0])
    nat_pairs = np.loadtxt("%s/native_contacts.ndx" % dir, skiprows=1, dtype=int) - 1
    n_native_pairs = nat_pairs.shape[0]

    pairs = np.loadtxt("%s/pairwise_params" % dir, usecols=(0,1))[2*n_native_pairs + 1::2] - 1
    r0 = np.loadtxt("%s/pairwise_params" % dir, usecols=(4,))[2*n_native_pairs + 1::2]
    top = "%s/Native.pdb" % dir
    #widths = 0.05*np.ones(len(r0), float)
    widths = 0.05

    r0_cont = r0 + 0.1
    Atanhsum_obs = observables.TanhContactSum(top, pairs, r0_cont, widths)
    Atanh = observables.calculate_observable(trajfiles, Atanhsum_obs, saveas="Atanh_0_05.dat")
