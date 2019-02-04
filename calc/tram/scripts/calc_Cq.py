import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")

import mdtraj as md
import pyemma.coordinates as coor
from pyemma.coordinates.data.featurization.misc import CustomFeature

import simulation.calc.tram.util as util
import simulation.calc.transits as transits

def dih_cosine(traj, dih_idxs, phi0):
    phi = md.compute_dihedrals(traj, dih_idxs)
    return np.cos(phi - phi0)

def get_q_featurizer(topfile, pairwise_file, n_native_pairs):

    # define configuration space using contact and dihedral features
    ref = md.load(topfile)
    dihedral_idxs = np.array([ [i, i + 1, i + 2, i + 3] for i in range(ref.n_residues - 3) ])
    phi0 = md.compute_dihedrals(ref, dihedral_idxs)

    feat = coor.featurizer(topfile)
    feat, feature_info = util.sbm_contact_features(feat, pairwise_file, n_native_pairs, skip_nn=1)

    feat.add_custom_feature(CustomFeature(dih_cosine, dihedral_idxs, phi0, dim=len(dihedral_idxs)))

    feature_info["dihedrals"] = dihedral_idxs
    feature_info["phi0"] = phi0

    return feat, feature_info

def calculate_covq_vs_tau_for_dwells(state, n_native_pairs, pairwise_file, lagtimes, recalc):

    if state == "native":
        label = "N"
    elif state == "unfolded":
        label = "U"
    else:
        raise IOError("State needs to be native or unfolded")

    # get unfolded state definition
    os.chdir("Qtanh_0_05_profile")
    with open("T_used.dat") as fin:
        T_used = float(fin.read())
    minima = np.loadtxt("minima.dat")
    U = minima.min()/float(n_native_pairs)
    N = minima.max()/float(n_native_pairs)
    os.chdir("..")

    trajfiles = [ "T_{:.2f}_{}/traj.xtc".format(T_used, x) for x in [1,2,3] ]
    topfile = trajfiles[0].split("/")[0] + "/ref.pdb"
    T = T_used

    # define configuration space using contact and dihedral features
    ref = md.load(topfile)
    dihedral_idxs = np.array([ [i, i + 1, i + 2, i + 3] for i in range(ref.n_residues - 3) ])
    phi0 = md.compute_dihedrals(ref, dihedral_idxs)

    feat = coor.featurizer(topfile)
    feat, feature_info = util.sbm_contact_features(feat, pairwise_file, n_native_pairs, skip_nn=1)

    feat.add_custom_feature(CustomFeature(dih_cosine, dihedral_idxs, phi0, dim=len(dihedral_idxs)))

    feature_info["dihedrals"] = dihedral_idxs
    feature_info["phi0"] = phi0
    n_dim = len(feat.describe())
    
    if not os.path.exists("hidhdim_Cq_{}_{:.2f}".format(label, T)):
        os.mkdir("hidhdim_Cq_{}_{:.2f}".format(label, T))

    if not os.path.exists("hidhdim_Cq_{}_{:.2f}/avg_q.npy".format(label, T)):
        import calc_avg_q
        calc_avg_q.calculate_avg_q_for_dwells(state, pairwise_file, n_native_pairs)

    avg_q = np.load("hidhdim_Cq_{}_{:.2f}/avg_q.npy".format(label, T))

    # calculate the covariance matrices for all degrees of freedom
    for n in range(len(lagtimes)):
        if os.path.exists("hidhdim_Cq_{}_{:.2f}/C0_{}.npy".format(label, T, lagtimes[n])) and not recalc:
            continue
        else:
            print "Lag = ", lagtimes[n]
            all_C0 = []
            all_Ctau = []
            all_Ntot = []
            # loop over trajectories
            for m in range(len(trajfiles)):
                # load reaction coordinate
                xtraj = np.load("{}/Qtanh_0_05.npy".format(os.path.dirname(trajfiles[m])))/float(n_native_pairs)
                
                dtraj = np.zeros(xtraj.shape[0], int)
                dtraj[xtraj <= U] = 0
                dtraj[xtraj >= N] = 2
                dtraj[(xtraj > U) & (xtraj < N)] = 1
                dwellsU, dwellsN, transitsUN, transitsNU = transits.partition_dtraj(dtraj, 0, 2)
                if label == "N":
                    dwells = dwellsN
                elif label == "U":
                    dwells = dwellsU

                traj = md.load(trajfiles[m], top=topfile)

                if len(dwells) == 0:
                    continue

                useable = (dwells[:,1] > 2*lagtimes[n]) & (dwells[:,1] > 5)
                if len(useable) == 0:
                    continue

                temp_dwells = dwells[useable]
                Ntot = float(np.sum(temp_dwells[:,1]))

                # calculate covariance matrix for this trajectory
                traj_C_tau = np.zeros((n_dim, n_dim), float)
                traj_C0 = np.zeros((n_dim, n_dim), float)
                for i in range(len(temp_dwells)):
                    start, length = temp_dwells[i]
                    chunk = traj[start:start + length]
                    cov_obj = coor.covariance_lagged(feat.transform(chunk), lag=lagtimes[n], remove_constant_mean=avg_q)
                    traj_C_tau += (length/Ntot)*cov_obj.cov_tau
                    traj_C0 += (length/Ntot)*cov_obj.cov

                all_Ntot.append(Ntot)
                all_Ctau.append(traj_C_tau) 
                all_C0.append(traj_C0) 

            avgC_tau = np.zeros((n_dim, n_dim), float)
            avgC0 = np.zeros((n_dim, n_dim), float)
            N = np.sum(all_Ntot)
            if N > 0:
                for i in range(len(trajfiles)):
                    avgC_tau += (all_Ntot[i]/float(N))*all_Ctau[i]
                    avgC0 += (all_Ntot[i]/float(N))*all_C0[i]

            np.save("hidhdim_Cq_{}_{:.2f}/C_tau_{}.npy".format(label, T, lagtimes[n]), avgC_tau)
            np.save("hidhdim_Cq_{}_{:.2f}/C0_{}.npy".format(label, T, lagtimes[n]), avgC0)


def calculate_covq_vs_tau(state, topfile, trajfiles, n_native_pairs, pairwise_file, T, lagtimes, recalc):

    if state == "native":
        label = "N"
    elif state == "unfolded":
        label = "U"
    else:
        raise IOError("State needs to be native or unfolded")

    # prep features 
    feat, feature_info = get_q_featurizer(topfile, pairwise_file, n_native_pairs)
    traj_inp = coor.source(trajfiles, features=feat)

    if not os.path.exists("hidhdim_Cq_{}_{:.2f}".format(label, T)):
        os.mkdir("hidhdim_Cq_{}_{:.2f}".format(label, T))

    if not os.path.exists("hidhdim_Cq_{}_{:.2f}/avg_q.npy".format(label, T)):
        # calculate
        import calc_avg_q
        calc_avg_q.calculate_avg_q_for_trajs(state, T, trajfiles, pairwise_file, topfile, n_native_pairs)

    avg_q = np.load("hidhdim_Cq_{}_{:.2f}/avg_q.npy".format(label, T))

    n_dim = len(feat.describe())
    #np.save("hidhdim_Cq_{}_{:.2f}/lagtime_tau.npy".format(T), 0.5*np.array(lagtimes))

    # calculate time-lagged covariance matrix for all degrees of freedom
    Cq = np.zeros(len(lagtimes), float)
    for n in range(len(lagtimes)):
        if os.path.exists("hidhdim_Cq_{}_{:.2f}/C0_{}.npy".format(label, T, lagtimes[n])) and not recalc:
            continue
        else:
            print "Lag = ", lagtimes[n]
            cov_obj = coor.covariance_lagged(traj_inp, lag=lagtimes[n], remove_constant_mean=avg_q)
            avgC0 = cov_obj.cov
            avgC_tau = cov_obj.cov_tau

            np.save("hidhdim_Cq_{}_{:.2f}/C0_{}.npy".format(label, T, lagtimes[n]), avgC0)
            np.save("hidhdim_Cq_{}_{:.2f}/C_tau_{}.npy".format(label, T, lagtimes[n]), avgC_tau)

            # the trace is the auto-correlation of phase space vector
            Cq[n] = np.trace(avgC_tau)/np.trace(avgC0)
            if Cq[n] < 0:
                print "  ** Negative eigenvalues" 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pairwise_params_file")
    parser.add_argument("n_native_pairs", type=int)
    parser.add_argument("T", type=float)
    parser.add_argument("state", type=str)
    parser.add_argument("--lagtimes", 
            default=[1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 
                50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 
                500, 1000, 1500, 2000, 3000, 5000, 10000, 20000],
            nargs="+", type=int)
    parser.add_argument("--only_dwells", action="store_true")
    parser.add_argument("--recalc", action="store_true")

    args = parser.parse_args()
    n_native_pairs = args.n_native_pairs
    pairwise_file = args.pairwise_params_file
    T = args.T
    state = args.state
    lagtimes = args.lagtimes
    only_dwells = args.only_dwells
    recalc = args.recalc


    import time
    starttime = time.time()

    if only_dwells:
         calculate_covq_vs_tau_for_dwells(state, n_native_pairs, pairwise_file, lagtimes, recalc)
    else:
        trajfiles = glob.glob("T_{:.2f}_*/traj.xtc".format(T))
        topfile = glob.glob("T_*/ref.pdb")[0]
        calculate_covq_vs_tau(state, topfile, trajfiles, n_native_pairs, pairwise_file, T, lagtimes, recalc)

    endtime = time.time()
    print "took: {} min".format((endtime - starttime)/60.)

