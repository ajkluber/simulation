import os
import pickle
import numpy as np

import pymbar

global KB
KB = 0.0083145

def get_organized_temps(tempsfile):
    """Get directory names by temperature"""
    with open(tempsfile, "r") as fin:
        temperature_dirs = fin.read().split()

    organized_temps = {}
    for i in range(len(temperature_dirs)):
        temp_dir = temperature_dirs[i]
        temp_T = float(temp_dir.split("_")[1])
        if not temp_T in organized_temps.keys():
            organized_temps[temp_T] = [temp_dir]
        else:
            organized_temps[temp_T].append(temp_dir)

    return organized_temps

def get_mbar_multi_temp(tempsfile, n_interpolate, engfile="Etot.dat", usecols=(1,)):

    organized_temps = get_organized_temps(tempsfile)

    print "loading energies"
    E, u_kn, N_k, beta = get_energies_ukn(organized_temps, n_interpolate=n_interpolate, engfile=engfile, usecols=usecols)

    f_k = None
    # calculate mbar object
    if os.path.exists("mbar/mbar.pkl"):
        # loading the pre-calculated f_k speeds recalculation
        with open("mbar/mbar.pkl", "rb") as fhandle:
            mbar_pkl = pickle.load(fhandle)
            pre_N_k = mbar_pkl["N_k"]
            pre_f_k = mbar_pkl["f_k"]
            pre_beta = mbar_pkl["beta"]
        if len(beta) == len(pre_beta):
            if np.allclose(pre_beta, beta):
                print "using precalculated f_k"
                f_k = pre_f_k
    print "solving mbar"
    mbar = pymbar.MBAR(u_kn, N_k, initial_f_k=f_k)

    if not os.path.exists("mbar/mbar.pkl"):
        save_mbar_info(mbar, beta)
    return mbar, beta, E, u_kn, N_k

def get_energies_ukn(organized_temps, n_interpolate=5, engfile="Etot.dat", usecols=(1,)):
    """Get energies from directories and interpolate"""

    sorted_temps = organized_temps.keys()
    sorted_temps.sort()

    E_sims = []
    N_k = []
    beta = []
    full_temps = []
    for i in range(len(sorted_temps)):
        T = sorted_temps[i]
        dirs_for_temp = organized_temps[T]
        temp_N = 0
        for n in range(len(dirs_for_temp)):
            E = np.loadtxt("{}/{}".format(dirs_for_temp[n], engfile), usecols=usecols)
            E_sims.append(E)
            temp_N += len(E)

        N_k.append(temp_N)
        beta.append(1./(KB*T))

        # add interpolated temperatures between this temperature and the next.
        if i < (len(sorted_temps) - 1):
            interp_T = np.linspace(T, sorted_temps[i + 1], n_interpolate + 2)[1:-1]
            for k in range(n_interpolate):
                beta.append(1./(KB*interp_T[k]))
                N_k.append(0)

    E = np.concatenate(E_sims)
    N_k = np.array(N_k)
    beta = np.array(beta)

    # dimensionless energy at all thermodynamic states
    u_kn = np.zeros((len(beta),len(E)), float)
    for k in range(len(beta)):
        u_kn[k,:] = beta[k]*E

    return E, u_kn, N_k, beta

def save_mbar_info(mbar, beta):
    """Save information needed to reconstruct mbar"""

    if not os.path.exists("mbar"):
        os.mkdir("mbar")
    os.chdir("mbar")

    mbar_info = {}
    mbar_info["f_k"] = mbar.f_k
    mbar_info["N_k"] = mbar.N_k
    mbar_info["beta"] = beta

    with open("mbar.pkl", "wb") as fhandle:
        pickle.dump(mbar_info, fhandle, protocol=pickle.HIGHEST_PROTOCOL)

    os.chdir("..")

