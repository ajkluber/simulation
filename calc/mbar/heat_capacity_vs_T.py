import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import util

global KB
KB = 0.0083145

def calculate_Cv(tempsfile, n_interpolate=5, n_extrapolate=0, display=False,
        engfile="Etot.dat", usecols=(1,), long=False):
    """Calculate heat capacity curve as a function of temperature
    
    Parameters
    ----------
    tempsfile : str
        Filename that holds directory names. Directories are assumed to be
        named as T_###.##_0 where the pound signs (#) indicate the value of the
        temperature.
    n_interpolate : int
        Number of points to interpolate between given temperatures. The results
        of mbar tend to be smoother when using some interpolation.
    display : bool
        If true: display the plot to the screen. Else just save.
    """
    mbar, beta, E, u_kn, N_k = util.get_mbar_multi_temp(tempsfile,
            n_interpolate, n_extrapolate=n_extrapolate,
            engfile=engfile, usecols=usecols)

    print "calculating Cv"
    U, dU = mbar.computeExpectations(E, compute_uncertainty=False)
    U2, dU2 = mbar.computeExpectations(E**2, compute_uncertainty=False)
    Cv = KB*beta*beta*(U2 - U**2)

    save_cv_and_plot(beta, Cv, display=display, long=long)

def save_cv_and_plot(beta, Cv, display=False, long=False):
    """Save heat capacity calculation and plot
    
    Parameters
    ----------
    beta : np.ndarray (K)
        The inverse temperature
    Cv : np.ndarray (K)
        The heat capacity as a function of temperature.
    display : bool, opt.
        If true, display the plot to the screen.
    """
    if long:
        if not os.path.exists("long_mbar_Cv"):
            os.mkdir("long_mbar_Cv")
        os.chdir("long_mbar_Cv")
    else:
        if not os.path.exists("mbar_Cv"):
            os.mkdir("mbar_Cv")
        os.chdir("mbar_Cv")

    # save heat capacity curve and
    T = 1./(beta*KB)

    mbar_Tf = T[np.argwhere(Cv == np.max(Cv))[0][0]]
    with open("Tfguess", "w") as fhandle:
        print "mbar estimates Tf as: {:.2f}".format(mbar_Tf)
        fhandle.write("{:.2f}".format(mbar_Tf))

    np.savetxt("cv.dat", Cv)
    np.savetxt("T.dat", T)

    if not display:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(T, Cv)
    plt.xlabel("temperature (K)")
    plt.ylabel("heat capacity")
    plt.savefig("cv.pdf")
    plt.savefig("cv.png")
    os.chdir("..")

    if display:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tempsfile")
    parser.add_argument("--n_interpolate", default=5, type=int)
    parser.add_argument("--n_extrapolate", default=0, type=int)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    calculate_Cv(args.tempsfile, n_interpolate=args.n_interpolate, n_extrapolate=args.n_extrapolate, display=args.display)
