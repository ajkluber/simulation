import os
import numpy as np
import scipy.optimize
import scipy.special

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dt = 0.5 
    with open("Qtanh_0_05_profile/T_used.dat", "r") as fin:
        T_used = float(fin.read())

    os.chdir("Qtanh_0_05_Kramers")

    varQ = np.loadtxt("T_{:.2f}_Kramers.dat".format(T_used))[0]

    acf = np.load("T_{:.2f}_acf.npy".format(T_used))
    time = dt*np.arange(len(acf))

    if os.path.exists("T_{:.2f}_acf_std.npy".format(T_used)):
        acf_std = np.load("T_{:.2f}_acf_std.npy".format(T_used))
    else:
        acf_std = np.ones(len(acf), float)

    
    scale = 10.
    mono_exp = lambda t, c1, t1: c1*np.exp(-t/(t1/scale))
    bi_exp = lambda t, c1, c2, t1, t2: c1*np.exp(-t/(t1/scale)) + c2*np.exp(-t/(t2/scale))
    tri_exp = lambda t, c1, c2, c3, t1, t2, t3: c1*np.exp(-t/(t1/scale)) + c2*np.exp(-t/(t2/scale)) + c3*np.exp(-t/(t3/scale))
    stretch_exp = lambda t, c1, beta, t1: c1*np.exp(-((t/(t1/scale))**beta))

    models = [mono_exp, bi_exp, stretch_exp]
    model_names = ["mono_exp", "bi_exp", "stretch_exp"]
    n_model_params = [2., 4., 3.]

    p0_mono = np.array([0.8, 1])
    p0_bi = np.array([0.5, 0.5, 0.5, 1])
    p0_stretch = np.array([0.9, 0.5, 1])
    p0_tri = np.array([0.3, 0.3, 0.3, 0.5, 1, 1])
    p0_models = [p0_mono, p0_bi, p0_stretch] 

    #try a couple values for long timescale
    guess_long_t = np.linspace(20, 1000, 100)

    all_chi2 = []
    all_popts = []
    all_fits = []
    for i in range(len(guess_long_t)): 
        trial_chi2 = []
        trial_popts = []
        trial_fits = []
        for n in range(len(models)):
            p0 = p0_models[n]
            p0[-1] = 0.8*guess_long_t[i]
            temp_popt, temp_pcov = scipy.optimize.curve_fit(models[n], time, acf, p0=p0, sigma=acf_std)
            acf_fit = models[n](time, *temp_popt)
            chi2_temp = np.sum(((acf_fit - acf)/acf_std)**2)/(len(acf) - n_model_params[n])

            trial_chi2.append(chi2_temp)
            trial_popts.append(temp_popt)
            trial_fits.append(acf_fit)
        all_chi2.append(trial_chi2)
        all_popts.append(trial_popts)
        all_fits.append(trial_fits)

    # take the fit with the minimum chi2.
    all_chi2 = np.array(all_chi2)
    idx1, idx2 = np.argwhere(all_chi2 == np.min(all_chi2))[0]

    if model_names[idx2] == "stretch_exp" and all_popts[idx1][idx2][-1] > 1:
        # ignore stretched exponential if params unreasonable
        idx1, idx2 = np.argwhere(all_chi2[:, :-1] == np.min(all_chi2[:, :-1]))[0]

    # calculate average relaxation timescale 
    popt = all_popts[idx1][idx2]


    if model_names[idx2] == "mono_exp":
        c1, t1 = popt
        t1 /= scale
        avg_tau_r = c1*t1
        parm_text = r"$c_1 = {:.2f}$ $t_1 = {:.2f}$".format(c1, t1)
    elif model_names[idx2] == "bi_exp":
        c1, c2, t1, t2 = popt
        t1 /= scale
        t2 /= scale
        avg_tau_r = c1*t1 + c2*t2
        parm_text = r"$c_1 = {:.2f}$ $t_1 = {:.2f}$ $c_2 = {:.2f}$ $t_2 = {:.2f}$".format(c1, t1, c2, t2)
    elif model_names[idx2] == "stretch_exp":
        coeff, tau_K, beta = popt
        tau_K /= scale
        avg_tau_r = coeff*(tau_K/beta)*scipy.special.gamma(1./beta)
        parm_text = r"$c = {:.2f}$ $t_K = {:.2f}$ $\beta = {:.2f}$".format(coeff, tau_K, beta)

    acf_fit = all_fits[idx1][idx2]
    #modelsall_popts[idx1][idx2]

    Dfit = varQ/avg_tau_r
    print model_names[idx2], avg_tau_r, Dfit

    #data = np.array([ popt[x] for x in range(len(popt))] + [Dfit])
    np.save("T_{:.2f}_fit_popts.npy".format(T_used), popt)
    np.save("T_{:.2f}_fit_D.npy".format(T_used), np.concatenate([popt, np.array([Dfit])]))

    plt.figure()
    if not np.allclose(acf_std, np.ones(len(acf), float)):
        plt.errorbar(time, acf, yerr=acf_std)
    else:
        plt.plot(time, acf)

    plt.plot(time, acf_fit, 'k--')
    plt.title(parm_text)
    plt.legend(loc=1)
    plt.xlabel("Time")
    plt.ylabel("ACF")
    #plt.savefig("acf_with_biexp.pdf")
    plt.savefig("acf_with_fit.pdf")

    os.chdir("..")
