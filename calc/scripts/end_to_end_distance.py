import os
import numpy as np

import simulation.calc.observables as observables

if __name__ == "__main__":
    trajfiles = [ x.rstrip("\n") for x in open("ticatrajs","r").readlines() ]
    dir = os.path.dirname(trajfiles[0])
    nat_pairs = np.loadtxt("%s/native_contacts.ndx" % dir, skiprows=1, dtype=int) - 1
    n_native_pairs = nat_pairs.shape[0]
    top = "%s/Native.pdb" % dir

    with open(top, "r") as fin:
        n_residues = len(fin.readlines()) - 1

    end_pairs = np.array([[0, n_residues - 1]])

    r_obs = observables.Distances(top,end_pairs)
    r = np.concatenate(observables.calculate_observable(trajfiles, r_obs))

    # Get probability density of end-to-end distance of the unfolded state
    qtanh = np.concatenate([ np.loadtxt("%s/Qtanh_0_05.dat" % os.path.dirname(x)) for x in trajfiles ])
    minima = np.loadtxt("Qtanh_0_05_profile/minima.dat")[0]
    U = minima + 0.1*n_native_pairs

    n, bins = np.histogram(r[qtanh < U],bins=40,density=True)
    mid_bin = 0.5*(bins[1:] + bins[:-1])

    if not os.path.exists("r1N_distribution"):
        os.mkdir("r1N_distribution")

    os.chdir("r1N_distribution")
    np.savetxt("r1N_vs_bin.dat", n)
    np.savetxt("mid_bin.dat", mid_bin)
    os.chdir("..")
