import os
import argparse
import numpy as np

from pyemma.coordinates.acf import acf

import simulation.calc.transits as transits

def get_U_trajs(coordfile, n_native_pairs, min_len=100):
    coordname = coordfile.split(".")[0]

    if os.path.exists("{}_profile/T_used.dat".format(coordname)):
        os.chdir("{}_profile".format(coordname))
        with open("T_used.dat") as fin:
            T_used = float(fin.read())
        minima = np.loadtxt("minima.dat")
        U = minima.min()/n_native_pairs
        N = minima.max()/n_native_pairs
        os.chdir("..")
    else:
        raise IOError("No T_used.dat found.")

    xtrajs = [ np.load("T_{:.2f}_{}/{}".format(T_used, x, coordfile))/n_native_pairs for x in [1,2,3] ]

    # get coordinate when dwelling in unfolded state.
    x_U = []
    for i in range(len(xtrajs)):
        xtraj =  xtrajs[i]
        dtraj = np.zeros(xtraj.shape[0], int)
        dtraj[xtraj <= U] = 0
        dtraj[xtraj >= N] = 2
        dtraj[(xtraj > U) & (xtraj < N)] = 1
        dwellsU, dwellsN, transitsUN, transitsNU = transits.partition_dtraj(dtraj, 0, 2)

        for j in range(len(dwellsU)):
            chunk_len = dwellsU[j,1] 
            if chunk_len > min_len:
                start_idx = dwellsU[j,0]
                x_U.append(xtraj[start_idx: start_idx + chunk_len])

    return x_U

def calculate_D_from_acf(x_U, dt, max_lag=500):
    """Calculate diffusion coefficient in unfolded state
    
    Parameters
    ----------
    x_U : list of arrays
        Reaction coordinate trajectories in the unfolded state.
    dt : float
        Simulation timestep.
    max_lag : int, opt.
        Maximum lagtime to calculate autocorrelation function
    """

    # calculate autocorrelation function (acf) of reaction coordinate in 
    # unfolded state
    acf_xU = acf(x_U, max_lag=max_lag)

    var_xU = np.var(np.concatenate(x_U))
    tau_x = np.sum(acf_xU)*dt
    D_U = var_xU/tau_x

    return acf_xU[:,0], var_xU, tau_x, D_U

def calculate_Kramers_tau(tau_x, D_U, n_native_pairs):
    """Calculate diffusion coefficient in unfolded state"""
    # calculate the Kramer's law mfpt doing a double integral over the
    # free energy profile, assuming a constant diffusion coefficient
    # using D in the unfolded state

    os.chdir("Qtanh_0_05_profile")
    with open("T_used.dat") as fin:
        T_used = float(fin.read())

    minima = np.loadtxt("minima.dat")
    U = minima.min()/n_native_pairs
    N = minima.max()/n_native_pairs

    style_1 = "T_{:.2f}".format(T_used)
    style_2 = "T_{:.1f}".format(T_used)
    if os.path.exists(style_1 + "_F.dat"):
        F = np.loadtxt(style_1 + "_F.dat")
        x_mid_bin = np.loadtxt(style_1 + "_mid_bin.dat".format(T_used))/n_native_pairs
    elif os.path.exists(style_2 + "_F.dat"):
        F = np.loadtxt(style_2 + "_F.dat")
        x_mid_bin = np.loadtxt(style_2 + "_mid_bin.dat".format(T_used))/n_native_pairs
    else:
        raise IOError("no free energy profile!")

    os.chdir("..")

    dx = x_mid_bin[1] - x_mid_bin[0]

    lower_idx = np.argmin((x_mid_bin - U)**2)
    upper_idx = np.argmin((x_mid_bin - N)**2)
    tau_K_integral = 0 
    for q in range(lower_idx, upper_idx + 1):
        for q_prime in range(len(F)):
            tau_K_integral += dx*dx*np.exp(F[q] - F[q_prime])/D_U

    # use simplified formula that assumes curvature equal
    TS_idx = np.argmax(F[lower_idx:upper_idx + 1])
    dF_dagg = F[TS_idx] - F[lower_idx]
    tau_K_simple = 2*np.pi*tau_x*np.exp(dF_dagg)

    return tau_K_integral, tau_K_simple

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_native_pairs", 
            type=float, 
            help="Numer of native contacts")

    args = parser.parse_args()
    n_native_pairs = float(args.n_native_pairs)

    dt = 0.5 # ps
    kb = 0.0083145 # kJ / (mol K)
    coordfile = "Qtanh_0_05.npy"
    with open("Qtanh_0_05_profile/T_used.dat") as fin:
        T_used = float(fin.read())

    # get reaction coordinate in U state
    x_U = get_U_trajs(coordfile, n_native_pairs)

    # diffusion coefficient from the autocorrelation of Q in unfolded state.
    # See Socci ''diffusive dynamics...''; Hummer ''Position-dependent diffusion...''
    acf_xU, var_xU, tau_x, D_U = calculate_D_from_acf(x_U, dt)

    # calculate Kramer's mean-first passage time
    tau_K_integral, tau_K_simple = calculate_Kramers_tau(tau_x, D_U, n_native_pairs)
    #print tau_K_integral, tau_K_simple

    #import matplotlib.pyplot as plt
    #plt.plot(acf_xU)
    #plt.xlabel(r"lagtime $\tau$")
    #plt.ylabel(r"ACF $\langle x_{t}\cdot x_{t + \tau}\rangle$")
    #plt.show()

    if not os.path.exists("Qtanh_0_05_Kramers"):
        os.mkdir("Qtanh_0_05_Kramers")
    os.chdir("Qtanh_0_05_Kramers")
    np.savetxt("T_{:.2f}_Kramers.dat".format(T_used), np.array([var_xU, tau_x, D_U, tau_K_integral, tau_K_simple]))
    np.save("T_{:.2f}_acf.npy".format(T_used), acf_xU)
    os.chdir("..")
