import os
import glob
import argparse
import numpy as np
import scipy.optimize
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import simulation.calc.transits as transits

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--p0", type=float, default=[0.3, 6., 4., 1.], nargs="+", help="Initial parameters")
    parser.add_argument("--p0", type=float, default=[0.3, 6., 4., 1.], nargs="+", help="Initial parameters")

    args = parser.parse_args()
    p0 = args.p0

    coord1 = "Qtanh_0_05.npy"
    #coord2 = "Rg.npy"
    coord2 = "r1N.npy"
    coord1_name = coord1.split(".")[0]
    coord2_name = coord2.split(".")[0]

    # determine which temperature is closest to the folding temperature
    with open("{}_profile/T_used.dat".format(coord1_name), "r") as fin:
        T_used = float(fin.read())

    minima = np.loadtxt("{}_profile/minima.dat".format(coord1_name))
    U = minima.min()
    N = minima.max()

    Tdirs = glob.glob("T_{:.2f}_1".format(T_used)) + glob.glob("T_{:.2f}_2".format(T_used)) + glob.glob("T_{:.2f}_3".format(T_used))

    coord2_trajs = []
    for i in range(len(Tdirs)):
        #print Tdirs[i]
        os.chdir(Tdirs[i])
        if coord1.endswith(".npy"):
            x = np.load(coord1)
        else:
            x = np.loadtxt(coord1)

        if coord2.endswith(".npy"):
            y = np.load(coord2)
        else:
            y = np.loadtxt(coord2)
        
        #print x.shape
        dtraj = np.zeros(x.shape[0], int)
        dtraj[x <= U] = 0
        dtraj[x >= N] = 2
        dtraj[(x > U) & (x < N)] = 1

        # calculate transits
        dwellsU, dwellsN, transitsUN, transitsNU = transits.partition_dtraj(dtraj, 0, 2)

        os.chdir("..")
        
        for n in range(len(dwellsU)):
            start, length = dwellsU[n]
            coord2_trajs.append(y[start:start + length])
    coord2_traj = np.concatenate(coord2_trajs)

    savedir = coord1_name + "_" + "dist" + "_" + coord2_name
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    os.chdir(savedir)

    n, bins = np.histogram(coord2_traj, bins=40, density=True)
    mid_bin = 0.5*(bins[1:] + bins[:-1])
    np.save("mid_bin.npy", mid_bin)
    np.save(coord2_name + "_dist.npy", n)

    scale = 10.
    def saw_r1N(r, A, nu, R, alpha):
        gamma = 1.1615
        g_factor = (gamma - 1)/(nu/scale)
        delta = 1/(1 - nu/scale)
        return A*(4*np.pi/R)*((r/R)**(2+g_factor))*np.exp(-alpha*((r/R)**delta))

    #popt, pcov = scipy.optimize.curve_fit(saw_r1N, mid_bin, n, p0=(1, 1, 5, 1))
    #popt, pcov = scipy.optimize.curve_fit(saw_r1N, mid_bin, n, p0=(1, 1., 5, 1))
    popt, pcov = scipy.optimize.curve_fit(saw_r1N, mid_bin, n, p0=tuple(p0))

    #params = np.copy(popt)
    #params[1] *= scale
    np.save("saw_params.npy", popt)
    #print popt
    #print pcov

    plt.figure()
    plt.plot(mid_bin, n)
    plt.plot(mid_bin, saw_r1N(mid_bin, *popt), 'k--')
    plt.savefig(coord2_name + "_dist.pdf")
     
    os.chdir("..")


    

