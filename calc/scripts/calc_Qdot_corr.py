import os
import argparse
import numpy as np

import simulation.calc.transits as transits
import simulation.calc.util as util

from pyemma.coordinates.acf import acf

def calc_D_cold_temps(coordfile, n_native_pairs, dt, max_lag):
    """Calculate diffusion coefficient"""

    organized_temps = util.get_organized_temps("cold_temps")
    T = organized_temps.keys()
    T.sort()

    D_all = []
    for i in range(len(T)):
        xtrajs = [ np.load("{}/{}".format(tdir,coordfile))/float(n_native_pairs) for tdir in organized_temps[T[i]] ]
        acf_xU, var_xU, tau_x, D_U = transits.calculate_D_from_acf(xtrajs, dt, max_lag=max_lag)
         
        if not os.path.exists("Qtanh_0_05_Kramers"):
            os.mkdir("Qtanh_0_05_Kramers")
        os.chdir("Qtanh_0_05_Kramers")
        np.savetxt("T_{:.2f}_D.dat".format(T[i]), np.array([var_xU, tau_x, D_U]))
        np.save("T_{:.2f}_acf.npy".format(T[i]), acf_xU)
        os.chdir("..")

        D_all.append(D_U)
    return T, D_all


def dummy():
    import matplotlib.pyplot as plt

    organized_temps = util.get_organized_temps("cold_temps")
    T = organized_temps.keys()
    T.sort()

    D_all = []
    acf_all = []
    tau_all = []
    for i in range(len(T)):
        xtrajs = [ np.load("{}/{}".format(tdir,coordfile)) for tdir in organized_temps[T[i]] ]
        acf_xU, var_xU, tau_x, D_U = transits.calculate_D_from_acf(xtrajs, dt)
        acf_all.append(acf_xU)
        tau_all.append(tau_x)
        D_all.append(D_U)

        #plt.hist(np.concatenate(xtrajs), bins=100, label=str(T[i]))
        #plt.figure()
        #for n in range(len(xtrajs)):
        #    plt.plot(xtrajs[n] + 50*n)
    #plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_native_pairs", 
            type=float, 
            help="Numer of native contacts")

    parser.add_argument("--dt", 
            type=float,
            default=0.5,
            help="Timestep size in ps.")

    parser.add_argument("--max_lag", 
            type=int,
            default=500,
            help="Maximum lagtime in frames.")

    parser.add_argument("--cold_temps", 
            action="store_true",
            help="Calculate at colder temps")

    args = parser.parse_args()
    n_native_pairs = float(args.n_native_pairs)
    dt = args.dt
    max_lag = args.max_lag
    cold_temps = args.cold_temps
    coordfile = "Qtanh_0_05.npy"


    organized_temps = util.get_organized_temps("cold_temps")
    T = organized_temps.keys()
    T.sort()

    acf_all = []
    #for i in range(len(T)):
    for i in [0]:
        xtrajs = [ np.load("{}/{}".format(tdir,coordfile))/float(n_native_pairs) for tdir in organized_temps[T[i]] ]
        xdot_trajs = [ xtrajs[j][1:] - xtrajs[j][:-1] for j in range(len(xtrajs)) ]

        acf_xdot = acf(xdot_trajs, max_lag=max_lag)
         
        acf_all.append(acf_xdot)

