import os
import numpy as np

import simulation.calc.observables as observables

if __name__ == "__main__":
    trajfiles = [ x.rstrip("\n") for x in open("ticatrajs","r").readlines() ]
    dir = os.path.dirname(trajfiles[0])
    pairs = np.loadtxt("%s/native_contacts.ndx" % dir, skiprows=1, dtype=int) - 1
    n_native_pairs = pairs.shape[0]
    r0 = np.loadtxt("%s/pairwise_params" % dir, usecols=(4,))[1:2*n_native_pairs:2]
    top = "%s/Native.pdb" % dir
    widths = 0.05*np.ones(n_native_pairs, float)

    r0_cont = r0 + 0.1
    qtanhsum_obs = observables.TanhContactSum(top, pairs, r0_cont, widths)
    qtanh = observables.calculate_observable(trajfiles, qtanhsum_obs, saveas="Qtanh_0_05.dat")
