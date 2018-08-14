import os
import numpy as np
import scipy.optimize

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
        acd_std = np.ones(len(acf), float)

    scale = 10.
    mono_exp = lambda t, c1, t1: c1*np.exp(-t/(t1/scale))
    bi_exp = lambda t, c1, c2, t1, t2: c1*np.exp(-t/(t1/scale)) + c2*np.exp(-t/(t2/scale))
    tr_exp = lambda t, c1, c2, c3, t1, t2, t3: c1*np.exp(-t/(t1/scale)) + c2*np.exp(-t/(t2/scale)) + c3*np.exp(-t/(t3/scale))

    #try a couple values for long timescale
    guess_long_t = np.linspace(20, 1000, 100)

    all_chi2 = []
    all_popts = []
    for i in range(len(guess_long_t)): 
        p0_mo = (0.8, 0.5*guess_long_t[i])
        popt_mo, pcov_mo = scipy.optimize.curve_fit(mono_exp, time, acf, p0=p0_mo, sigma=acf_std)
        fit_mo = mono_exp(time, *popt_mo)
        chi2_mo = np.sum(((fit_mo - acf)/acf_std)**2)/(len(acf) - 2.)

        p0_bi = (0.5, 0.5, 0.5, guess_long_t[i])
        popt_bi, pcov_bi = scipy.optimize.curve_fit(bi_exp, time, acf, p0=p0_bi, sigma=acf_std)
        fit_bi = bi_exp(time, *popt_bi)
        chi2_bi = np.sum(((fit_bi - acf)/acf_std)**2)/(len(acf) - 4.)

        # for 1qlx three exponentials often can't find a solution
        #try:
        #    p0_tr = (0.3, 0.3, 0.3, 0.5, 0.5*guess_long_t[i], guess_long_t[i])
        #    popt_tr, pcov_tr = scipy.optimize.curve_fit(tr_exp, time, acf, p0=p0_tr, sigma=acf_std)
        #    fit_tr = tr_exp(time, *popt_tr)
        #    chi2_tr = np.sum(((fit_tr - acf)/acf_std)**2)/(len(acf) - 6.)
        #except RuntimeError:
        #    chi2_tr = "n/a"
        #print chi2_mo, chi2_bi, chi2_tr
        all_chi2.append([chi2_mo, chi2_bi])

        if popt_bi[3] < popt_bi[2]:
            new_popt_bi = (popt_bi[1], popt_bi[0], popt_bi[3], popt_bi[2])
            popt_bi = new_popt_bi

        all_popts.append([popt_mo, popt_bi])
        #print squared_error, popt

    all_chi2 = np.array(all_chi2)
    idx1, idx2 = np.argwhere(all_chi2 == np.min(all_chi2))[0]
    popt = all_popts[idx1][idx2]
    tau_r = popt[-1]/scale

    data = np.array([ popt[x] for x in range(len(popt))] + [varQ/tau_r])
    np.save("T_{:.2f}_biexp.npy".format(T_used), data)

    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.rm'] = 'serif'
    import matplotlib.pyplot as plt

    plt.figure()
    if not np.allclose(acf_std, np.ones(len(acf), float)):
        plt.errorbar(time, acf, yerr=acf_std)
    else:
        plt.plot(time, acf)

    if idx2 == 0:
        plt.plot(time, mono_exp(time, *popt), 'k--', label=r"$t_1 = {:.3e}$".format(popt[-1]/scale))
    else:
        plt.plot(time, bi_exp(time, *popt), 'k--', label=r"$t_1 = {:.3e}$  $t_2={:.3e}$".format(popt[-2]/scale, popt[-1]/scale))
    plt.legend(loc=1)
    plt.xlabel("Time")
    plt.ylabel("ACF")
    plt.savefig("acf_with_biexp.pdf")

    os.chdir("..")
