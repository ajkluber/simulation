import os
import glob
import argparse
import numpy as np

import simulation.calc.transits as transits
import simulation.calc.util as util

def calc_Kramers_tau_at_Tf(coordfile, n_native_pairs, dt, max_lag):
    """Calculate Kramers theory folding time and diffusion coefficient"""

    #import pdb 
    #pdb.set_trace()
    temp_dirs = glob.glob("T_*_1")
    if os.path.exists("Qtanh_0_05_profile/T_used.dat"):
        with open("Qtanh_0_05_profile/T_used.dat") as fin:
            T_used = float(fin.read())
    elif os.path.exists("T_used.dat"):
        with open("T_used.dat") as fin:
            T_used = float(fin.read())
    elif len(temp_dirs) > 0:
        T_used = float(temp_dirs[0].split("_")[1])
    else:
        raise IOError("Missing input file T_used.dat")

    # get reaction coordinate in U state
    x_U = transits.get_U_trajs(coordfile, n_native_pairs)

    # diffusion coefficient from the autocorrelation of Q in unfolded state.
    # See Socci ''diffusive dynamics...''; Hummer ''Position-dependent diffusion...''
    acf_xU, var_xU, tau_x, D_U, acf_std = transits.calculate_D_from_acf(x_U, dt, max_lag=max_lag)

    # calculate Kramer's mean-first passage time
    tau_K_integral, tau_K_simple = transits.calculate_Kramers_tau(tau_x, D_U, n_native_pairs)

    if not os.path.exists("Qtanh_0_05_Kramers"):
        os.mkdir("Qtanh_0_05_Kramers")
    os.chdir("Qtanh_0_05_Kramers")
    np.savetxt("T_{:.2f}_Kramers.dat".format(T_used), np.array([var_xU, tau_x, D_U, tau_K_integral, tau_K_simple]))
    np.save("T_{:.2f}_acf.npy".format(T_used), acf_xU)
    if len(acf_std) > 0:
        np.save("T_{:.2f}_acf_std.npy".format(T_used), acf_std)
    os.chdir("..")

def calc_D_cold_temps(coordfile, n_native_pairs, dt, max_lag):
    """Calculate diffusion coefficient"""

    organized_temps = util.get_organized_temps("cold_temps")
    T = organized_temps.keys()
    T.sort()

    D_all = []
    for i in range(len(T)):
        xtrajs = [ np.load("{}/{}".format(tdir,coordfile))/float(n_native_pairs) for tdir in organized_temps[T[i]] ]
        acf_xU, var_xU, tau_x, D_U, acf_std = transits.calculate_D_from_acf(xtrajs, dt, max_lag=max_lag)
         
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
        acf_xU, var_xU, tau_x, D_U, acf_std = transits.calculate_D_from_acf(xtrajs, dt)
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

    if cold_temps:
        T, D_all = calc_D_cold_temps(coordfile, n_native_pairs, dt, max_lag)
    else:
        calc_Kramers_tau_at_Tf(coordfile, n_native_pairs, dt, max_lag)
