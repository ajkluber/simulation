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

def calculate_avg_q_for_U_dwells(pairwise_file, n_native_pairs):

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

    # make featurizer
    feat, feature_info = get_q_featurizer(topfile, pairwise_file, n_native_pairs)

    n_dim = len(feat.describe())
    
    if not os.path.exists("hidhdim_Cq_U_{:.2f}".format(T)):
        os.mkdir("hidhdim_Cq_U_{:.2f}".format(T))

    all_avg_q = []
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

        traj = md.load(trajfiles[m], top=topfile)

        if len(dwellsU) == 0:
            break 

        # calculate covariance matrix for this trajectory
        traj_avg_q = np.zeros(n_dim, float)
        Ntot = 0.
        for i in range(len(dwellsU)):
            start, length = dwellsU[i]
            chunk = traj[start:start + length]
            traj_avg_q += np.sum(feat.transform(chunk), axis=0)
            Ntot += length
        all_Ntot.append(Ntot)
        all_avg_q.append(traj_avg_q/float(Ntot)) 

    avg_q = np.zeros(n_dim, float)
    N = np.sum(all_Ntot)
    if N > 0:
        for i in range(len(trajfiles)):
            avg_q += (all_Ntot[i]/float(N))*all_avg_q[i]

    np.save("hidhdim_Cq_U_{:.2f}/avg_q.npy".format(T), avg_q)

def calculate_avg_q_for_trajs(T, trajfiles, pairwise_file, topfile, n_native_pairs):

    # make featurizer
    feat, feature_info = get_q_featurizer(topfile, pairwise_file, n_native_pairs)
    n_dim = len(feat.describe())
    
    if not os.path.exists("hidhdim_Cq_U_{:.2f}".format(T)):
        os.mkdir("hidhdim_Cq_U_{:.2f}".format(T))

    all_avg_q = []
    all_Ntot = []
    # loop over trajectories
    for m in range(len(trajfiles)):
        # calculate average phase space vector for trajectory
        traj_avg_q = np.zeros(n_dim, float)
        Ntot = 0.
        for chunk in md.iterload(trajfiles[m], top=topfile):
            traj_avg_q += np.sum(feat.transform(chunk), axis=0)
            Ntot += chunk.n_frames
        all_Ntot.append(Ntot)
        all_avg_q.append(traj_avg_q/float(Ntot)) 

    avg_q = np.zeros(n_dim, float)
    N = np.sum(all_Ntot)
    if N > 0:
        for i in range(len(trajfiles)):
            avg_q += (all_Ntot[i]/float(N))*all_avg_q[i]

    np.save("hidhdim_Cq_U_{:.2f}/avg_q.npy".format(T), avg_q)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pairwise_params_file")
    parser.add_argument("n_native_pairs", type=int)
    parser.add_argument("T", type=float)
    parser.add_argument("--only_U_dwells", action="store_true")
    parser.add_argument("--recalc", action="store_true")

    args = parser.parse_args()
    n_native_pairs = args.n_native_pairs
    pairwise_file = args.pairwise_params_file
    T = args.T
    only_U_dwells = args.only_U_dwells


    import time
    starttime = time.time()

    if only_U_dwells:
        calculate_avg_q_for_U_dwells(pairwise_file, n_native_pairs)

    endtime = time.time()
    print "took: {} min".format((endtime - starttime)/60.)

