import os
import numpy as np

from pyemma.coordinates.acf import acf

def partition_dtraj(dtraj, state1, state2): 
    """ Partition the trajectory into transits bewteen states and intra-state dwells.

    Parameters
    ----------
    dtraj : np.ndarray, or list
        Discrete trajectory; timeseries of state-occupancies. State elements .
    
    state1 : int
        state to  

    state2 : int



    # discard initial frames that aren't in either state1 or state2.

    Returns
    -------
    dwells : list 
    # collect the first frame index and length of:
    #   - dwells in state1
    #   - dwells in state2
    #   - transits from state1 to state2
    #   - transits from state2 to state1


    """
    nframes =  len(dtraj)
    state_idx = {state1:0, state2:1}
    states = [state1, state2]
    
    # discard frames that are in transit region.
    discard_frames = 0
    for i in range(nframes):
        if dtraj[i] in states:
            prev_state = dtraj[i]
            from_state = dtraj[i]
            break
        else:
            discard_frames += 1

    # collect list of [start_idx, length] for dwells and transits
    dwells = [[], []]
    transits = [[],[]]

    dwell_len = 1
    dwell_start_idx = discard_frames
    for i in range(discard_frames, nframes):
        curr_state = dtraj[i]
        if curr_state not in states:
            # current frame is in transit region
            if prev_state not in states:
                # continuing transit
                transit_len += 1
            else:
                # starting potential transit
                transit_len = 1
                transit_start_idx = i
        else:
            # current frame is in state1 or state2 
            if prev_state == curr_state:
                # extend dwell
                dwell_len += 1
            elif (prev_state != curr_state) and (prev_state not in states):
                if from_state == curr_state:
                    # failed transit; extend dwell time
                    dwell_len += transit_len + 1
                else:
                    # successful transit! starting new dwell
                    transits[state_idx[from_state]].append([transit_start_idx, transit_len])
                    dwells[state_idx[from_state]].append([dwell_start_idx, dwell_len])
                    dwell_start_idx = i
                    dwell_len = 1
                    from_state = curr_state
            else:
                # transit of length 0! this should never happen.
                #raise IOError("This trajectory has transits of zero frames!")
                print "WARNING: This trajectory has transits of zero frames!"

        prev_state = curr_state
    dwells1 = np.array(dwells[0])
    dwells2 = np.array(dwells[1])
    transits12 = np.array(transits[0])
    transits21 = np.array(transits[1])
    return dwells1, dwells2, transits12, transits21

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
        dwellsU, dwellsN, transitsUN, transitsNU = partition_dtraj(dtraj, 0, 2)

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
    max_lag = np.min([max_lag, int(np.floor(np.percentile([ len(x_U[i]) for i in range(len(x_U)) ], 50)))])

    acf_xU = acf(x_U, max_lag=max_lag)
    N = len(x_U)
    if N > 10:
        idxs = np.random.permutation(np.arange(N))
        n_folds = 4
        size = N/n_folds 
        all_acf = []
        for i in range(n_folds):
            use_idxs = idxs[i*N/n_folds:(i + 1)*N/n_folds] 
            temp_xU = []
            for j in range(len(use_idxs)):
                temp_xU.append(x_U[use_idxs[j]])
            acf_temp = acf(temp_xU, max_lag=max_lag)
            all_acf.append(acf_temp[:,0])
        # what if 
        acf_std = np.std(np.array(all_acf), axis=0)
        acf_std[acf_std == 0] = 1e-8
    else:
        acf_std = []

    var_xU = np.var(np.concatenate(x_U))
    tau_x = np.sum(acf_xU)*dt
    D_U = var_xU/tau_x

    return acf_xU[:,0], var_xU, tau_x, D_U, acf_std

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
