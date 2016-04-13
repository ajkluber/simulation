import os
import numpy as np
import matplotlib.pyplot as plt

import mdtraj as md
import pymbar

import model_builder as mdb
import simulation.calc.observables as observables

def get_unbiased_energy(Q0_dirs):
    E_no_bias_all = []
    qtanh_all = []
    umb_params = []
    N_k_bias = []

    # Calculate the potential energy with the bias for each umbrella center.
    for i in range(n_biases):
        os.chdir(Q0_dirs[i])
        calculate_E_no_bias(E_no_bias_all, qtanh_all, umb_params, N_k_bias) 
        os.chdir("..")

    E_no_bias = np.concatenate(E_no_bias_all)
    qtanh = np.concatenate(qtanh_all)
    N_k = np.array(N_k_bias + [0])

    return E_no_bias, qtanh, umb_params, N_k

def calculate_E_no_bias(E_no_bias_all, qtanh_all, umb_params, N_k_bias):

    with open("umbrella_params", "r") as fin:
        params = fin.readline().split()
    qtanh_center = float(params[0])
    kumb = float(params[1])
    gamma = float(params[2])
    n_frames_toss = int(params[4])/1000
    umb_params.append([qtanh_center, kumb, n_frames_toss])

    if not os.path.exists("qtanh.npy"):
        pairs = np.loadtxt("umbrella_params", usecols=(0,1), dtype=int, skiprows=1) - 1
        r0 = np.loadtxt("umbrella_params", usecols=(2,), skiprows=1)

        widths = (2./gamma)*np.ones(len(pairs))

        qtanhsum_obs = observables.TanhContactSum("conf.gro", pairs, 1.2*r0, widths)
        qtanh_temp = observables.calculate_observable(["traj.xtc"], qtanhsum_obs)[0][n_frames_toss:]
        np.save("qtanh.npy", qtanh_temp)
    else:
        qtanh_temp = np.load("qtanh.npy")

    # collect needed info for this umbrella
    Ebias = 0.5*kumb*((qtanh_temp - qtanh_center)**2)
    Epot = np.loadtxt("Epot.dat", usecols=(1,))[n_frames_toss:]
    E_no_bias_all.append(Epot - Ebias)

    qtanh_all.append(qtanh_temp)

    N_k_bias.append(qtanh_temp.shape[0])

def get_mbar(E_no_bias, qtanh, umb_params, N_k):
    # calculate dimensionless energy for every frame in every thermodynamic
    # state (umbrella). The first state is the unbiased state.
    u_kn = np.zeros((n_biases + 1, E_no_bias.shape[0]))
    for i in range(n_biases):
        qtanh_center = umb_params[i][0]
        kumb = umb_params[i][1]
        Ebias_k = 0.5*kumb*((qtanh - qtanh_center)**2)
        u_kn[i, :] = beta*(E_no_bias + Ebias_k)
    u_kn[-1, :] = beta*E_no_bias

    mbar = pymbar.MBAR(u_kn, N_k)
    return mbar

def calculate_unbiased_profile(mbar, qtanh, n_bins):
    # histogram qtanh over all therm states
    n, bin_edges = np.histogram(qtanh, bins=n_bins)
    mid_bin = 0.5*(bin_edges[1:] + bin_edges[:-1])

    # calculate bin expectations for all therm states
    h_bin_avg = np.zeros((n_biases + 1, n_bins),float) 
    dh_bin_avg = np.zeros((n_biases + 1, n_bins),float) 
    for i in range(n_bins):
        h = ((qtanh > bin_edges[i]) & (qtanh <= bin_edges[i + 1])).astype(int)
        h_bin_avg[:,i], dh_bin_avg[:,i] = mbar.computeExpectations(h)

    # calculate free energy profile at all therm states
    pmf_bin_avg = np.nan*np.zeros((n_biases + 1, n_bins),float) 
    for i in range(n_biases + 1):
        not_too_small = h_bin_avg[i, :] > (1E-5)
        pmf_temp = -np.log(h_bin_avg[i, not_too_small])
        pmf_bin_avg[i, not_too_small] = pmf_temp - pmf_temp.min()

    return mid_bin, pmf_bin_avg

def plot_free_energy_profiles(mid_bin, pmf_bin_avg, Q0_dirs):
    # plot free energy profiles
    plt.figure()
    for i in range(n_biases + 1):
        not_nan = pmf_bin_avg[i, :] != np.nan
        if i == n_biases: 
            plt.plot(mid_bin[not_nan], pmf_bin_avg[i, not_nan], lw=3, label="unbiased")
        else:
            plt.plot(mid_bin[not_nan], pmf_bin_avg[i, not_nan], label=Q0_dirs[i])

    plt.xlabel("$Q_{tanh}$")
    plt.ylabel("Free energy (k$_B$T)")
    plt.xlim(0, 150)
    plt.legend()
    plt.savefig("mbar_Fvsqtanh.pdf", bbox_inches="tight")
    plt.savefig("mbar_Fvsqtanh.png", bbox_inches="tight")

    plt.show()

if __name__ == "__main__":
    KB_KJ_MOL = 0.0083145
    T = 130.95 
    beta = 1./(T*KB_KJ_MOL)
    n_bins = 40

    Q0_dirs = [ x.rstrip("\n") for x in open("umbrella_last","r").readlines() ]

    n_biases = len(Q0_dirs) 

    print "getting unbiased energies"
    E_no_bias, qtanh, umb_params, N_k = get_unbiased_energy(Q0_dirs)

    print "calculating mbar"
    mbar = get_mbar(E_no_bias, qtanh, umb_params, N_k)

    print "calculating unbiased free energy"
    mid_bin, pmf_bin_avg = calculate_unbiased_profile(mbar, qtanh, n_bins)
    
    print "plotting"
    plot_free_energy_profiles(mid_bin, pmf_bin_avg, Q0_dirs)

