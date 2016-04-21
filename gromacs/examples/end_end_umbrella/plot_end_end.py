import os
import numpy as np
import matplotlib.pyplot as plt

import mdtraj as md

if __name__ == "__main__":
    r0 = [ x.rstrip("\n") for x in open("umbrella_last", "r").readlines() ] 
    n_bins = 40

    plt.figure()
    for i in range(len(r0)):

        # calculate pmf for umbrella
        os.chdir(r0[i])

        if not os.path.exists("r1N.npy"):
            traj = md.load("traj.xtc", top="conf.gro")
            r1N =  md.compute_distances(traj, np.array([[0,57]]))
            np.save("r1N.npy", r1N)
        else:
            r1N = np.load("r1N.npy")
            
        n, bins = np.histogram(r1N, bins=n_bins)
        mid_bin = 0.5*(bins[1:] + bins[:-1])
        pmf = -np.log(n)
        pmf -= pmf.min()

        plt.plot(mid_bin, pmf, label="$r_0 = {}$".format(r0[i]))
        
        os.chdir("..")

    plt.xlim(0, 14)
    plt.ylim(0, 6)
    plt.legend()
    plt.xlabel("End-End distance $r_{1N}$ (nm)")
    plt.ylabel("Free Energy (k$_B$T)")
    plt.savefig("Fvsr1N.pdf", bbox_inches="tight")
    plt.savefig("Fvsr1N.png", bbox_inches="tight")
    plt.show()
