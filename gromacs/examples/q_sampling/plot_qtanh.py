import os
import numpy as np
import matplotlib.pyplot as plt

import mdtraj as md

import simulation.calc.observables as observables

if __name__ == "__main__":
    n_bins = 40

    Q0 = [ x.rstrip("\n") for x in open("umbrella_last", "r").readlines() ] 

    plt.figure()
    for i in range(len(Q0)):

        # calculate pmf for umbrella
        os.chdir(Q0[i])

        if not os.path.exists("qtanh.npy"):
            traj = md.load("traj.xtc", top="conf.gro")
            qtanh =  md.compute_distances(traj, np.array([[0,57]]))
            with open("umbrella_params", "r") as fin:
                params = fin.readline().split()
            q0 = float(params[0])
            kumb = float(params[1])
            gamma = float(params[2])
            n_frames_toss = int(params[4])/1000

            pairs = np.loadtxt("umbrella_params", usecols=(0,1), dtype=int, skiprows=1) - 1
            r0 = np.loadtxt("umbrella_params", usecols=(2,), skiprows=1)

            widths = (2./gamma)*np.ones(len(pairs))

            qtanhsum_obs = observables.TanhContactSum("conf.gro", pairs, 1.2*r0, widths)
            qtanh = observables.calculate_observable(["traj.xtc"], qtanhsum_obs)

            np.save("qtanh.npy", qtanh)
        else:
            qtanh = np.load("qtanh.npy")
            
        n, bins = np.histogram(qtanh, bins=n_bins)
        mid_bin = 0.5*(bins[1:] + bins[:-1])
        pmf = -np.log(n)
        pmf -= pmf.min()

        plt.plot(mid_bin, pmf, label="$Q_0 = {}$".format(Q0[i]))
        
        os.chdir("..")

    plt.xlim(0, 150)
    plt.ylim(0, 6)
    plt.legend()
    plt.xlabel("End-End distance $Q_{tanh}$")
    plt.ylabel("Free Energy (k$_B$T)")
    plt.savefig("Fvsqtanh.pdf", bbox_inches="tight")
    plt.savefig("Fvsqtanh.png", bbox_inches="tight")
    plt.show()

