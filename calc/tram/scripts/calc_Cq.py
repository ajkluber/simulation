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


def calculate_covq_vs_tau_for_U_dwells(n_native_pairs, lagtimes, recalc):

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
    T = trajfiles[0].split("_")[1]
    
    all_dwellsU = []
    for i in range(len(trajfiles)):
        # load reaction coordinate
        xtraj = np.load("T_{:.2f}_1/Qtanh_0_05.npy".format(T_used))/float(n_native_pairs)
        
        dtraj = np.zeros(xtraj.shape[0], int)
        dtraj[xtraj <= U] = 0
        dtraj[xtraj >= N] = 2
        dtraj[(xtraj > U) & (xtraj < N)] = 1
        dwellsU, dwellsN, transitsUN, transitsNU = transits.partition_dtraj(dtraj, 0, 2)
        all_dwellsU.append(dwellsU)


    # define configuration space using contact and dihedral features
    ref = md.load(topfile)
    dihedral_idxs = np.array([ [i, i + 1, i + 2, i + 3] for i in range(ref.n_residues - 3) ])
    phi0 = md.compute_dihedrals(ref, dihedral_idxs)

    feat = coor.featurizer(topfile)
    feat, feature_info = util.sbm_contact_features(feat, pairwise_file, n_native_pairs, skip_nn=1)

    feat.add_custom_feature(CustomFeature(dih_cosine, dihedral_idxs, phi0, dim=len(dihedral_idxs)))

    feature_info["dihedrals"] = dihedral_idxs
    feature_info["phi0"] = phi0

    lagtimes = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200]

    traj = md.load("T_{:.2f}_1/traj.xtc".format(T), top=topfile)

    if not os.path.exists("hidhdim_Cq_U_{:.2f}".format(T)):
        os.mkdir("hidhdim_Cq_U_{:.2f}".format(T))

    n_dim = len(feat.describe())
    #np.save("lagtime_tau.npy", 0.5*np.array(lagtimes))

    # calculate the TICA covariance matrices for all degrees of freedom
    for n in range(len(lagtimes)):
        if os.path.exists("hidhdim_Cq_U_{:.2f}/C0_{}.npy".format(T, lagtimes[n])) and not recalc:
            continue
        else:
            print "Lag = ", lagtimes[n]
            useable = dwellsU[:,1] > 2*lagtimes[n]
            if len(useable) == 0:
                continue
            temp_dwellsU = dwellsU[useable]
            Ntot = float(np.sum(temp_dwellsU[:,1]))

            avgC_tau = np.zeros((n_dim, n_dim), float)
            avgC0 = np.zeros((n_dim, n_dim), float)
            for i in range(len(temp_dwellsU)):
                start, length = temp_dwellsU[i]
                chunk = traj[start:start + length]
                temp_tica = coor.tica(feat.transform(chunk), lagtimes[n])
                avgC_tau += (length/Ntot)*temp_tica.cov_tau
                avgC0 += (length/Ntot)*temp_tica.cov

            np.save("hidhdim_Cq_U_{:.2f}/C_tau_{}.npy".format(T, lagtimes[n]), avgC_tau)
            np.save("hidhdim_Cq_U_{:.2f}/C0_{}.npy".format(T, lagtimes[n]), avgC0)


def calculate_covq_vs_tau(topfile, trajfiles, n_native_pairs, T, lagtimes, recalc):

    ref = md.load(topfile)
    dihedral_idxs = np.array([ [i, i + 1, i + 2, i + 3] for i in range(ref.n_residues - 3) ])
    phi0 = md.compute_dihedrals(ref, dihedral_idxs)

    feat = coor.featurizer(topfile)
    feat, feature_info = util.sbm_contact_features(feat, pairwise_file, n_native_pairs, skip_nn=1)

    feat.add_custom_feature(CustomFeature(dih_cosine, dihedral_idxs, phi0, dim=len(dihedral_idxs)))

    feature_info["dihedrals"] = dihedral_idxs
    feature_info["phi0"] = phi0

    traj_inp = coor.source(trajfiles, features=feat)

    if not os.path.exists("hidhdim_Cq_U_{:.2f}".format(T)):
        os.mkdir("hidhdim_Cq_U_{:.2f}".format(T))

    n_dim = len(feat.describe())
    np.save("hidhdim_Cq_U_{:.2f}/lagtime_tau.npy".format(T), 0.5*np.array(lagtimes))

    # calculate time-lagged covariance matrix for all degrees of freedom
    Cq = np.zeros(len(lagtimes), float)
    for n in range(len(lagtimes)):
        if os.path.exists("hidhdim_Cq_U_{:.2f}/C0_{}.npy".format(T, lagtimes[n])) and not recalc:
            continue
        else:
            print "Lag = ", lagtimes[n]
            cov_obj = coor.covariance_lagged(traj_inp, lag=lagtimes[n], stride=10)
            avgC0 = cov_obj.cov
            avgC_tau = cov_obj.cov_tau

            np.save("hidhdim_Cq_U_{:.2f}/C0_{}.npy".format(T, lagtimes[n]), avgC0)
            np.save("hidhdim_Cq_U_{:.2f}/C_tau_{}.npy".format(T, lagtimes[n]), avgC_tau)

            # the trace is the auto-correlation of phase space vector
            Cq[n] = np.trace(avgC_tau)/np.trace(avgC0)
            if Cq[n] < 0:
                print "  ** Negative eigenvalues" 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pairwise_params_file")
    parser.add_argument("n_native_pairs", type=int)
    parser.add_argument("T", type=float)
    parser.add_argument("--lagtimes", 
            default=[1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 
                50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 
                500, 1000, 1500, 2000, 3000, 5000, 10000, 20000],
            nargs="+", type=int)
    parser.add_argument("--only_U_dwells", action="store_true")
    parser.add_argument("--recalc", action="store_true")

    args = parser.parse_args()
    n_native_pairs = args.n_native_pairs
    pairwise_file = args.pairwise_params_file
    T = args.T
    lagtimes = args.lagtimes
    only_U_dwells = args.only_U_dwells
    recalc = args.recalc

    trajfiles = glob.glob("T_{:.2f}_*/traj.xtc".format(T))
    topfile = glob.glob("T_{:.2f}_*/ref.pdb".format(T))[0]

    if only_U_dwells:
         calculate_covq_vs_tau_for_U_dwells(n_native_pairs, lagtimes, recalc)
    else:
        calculate_covq_vs_tau(topfile, trajfiles, n_native_pairs, T, lagtimes, recalc)

