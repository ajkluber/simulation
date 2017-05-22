import os
import argparse
import numpy as np

import mdtraj as md
import pyemma.coordinates as coor

import simulation.calc.tram.util as util

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pairwise_params_file")
    parser.add_argument("n_native_pairs", type=int)

    args = parser.parse_args()
    n_native_pairs = args.n_native_pairs
    pairwise_file = args.pairwise_params_file

    n_qbins = 40
    topname = "ref.pdb"
    trajname = "traj.xtc"
    color_idxs = np.arange(n_qbins)/float(n_qbins)

    with open("Qtanh_0_05_profile/T_used.dat","r") as fin:
        T = float(fin.read())
    tempdirs = [ "T_{:.2f}_{}".format(T, x) for x in [1,2,3] ]

    topfile = tempdirs[0] + "/" + topname
    trajfiles = [ x + "/" + trajname for x in tempdirs ]

    qtrajs = [ np.load(x + "/Qtanh_0_05.npy") for x in tempdirs ]

    # add features
    non_feat = coor.featurizer(topfile)
    non_feat, feature_info = util.sbm_contact_features(non_feat, pairwise_file, n_native_pairs, skip_nn=1, nonnative_only=True)
    n_res = non_feat.topology.n_residues

    n_non_dim = feature_info['dim']
    non_pairs = feature_info['pairs']
    loop = non_pairs[:,1] - non_pairs[:,0]
    local_idxs = (loop <= 8)
    nonlocal_idxs = (loop > 8)

    P_q, qbins = np.histogram(np.concatenate(qtrajs), bins=n_qbins, density=True)
    P_q *= (qbins[1] - qbins[0])
    q_mid_bin = 0.5*(qbins[:-1] + qbins[1:])

    # calculate contacts as a function of Q.
    Alocal_vs_Q = np.zeros(n_qbins, float)
    Anonlocal_vs_Q = np.zeros(n_qbins, float)
    bin_counts = np.zeros(n_qbins)
    for i in range(len(trajfiles)):
        start_idx = 0
        for chunk in md.iterload(trajfiles[i], top=non_feat.topologyfile):
            idxs = np.arange(chunk.n_frames) + start_idx
            Ai = non_feat.transform(chunk)
            Alocal = np.sum(Ai[:, local_idxs], axis=1)
            Anonlocal = np.sum(Ai[:, nonlocal_idxs], axis=1)
            for n in range(n_qbins - 1):
                frames_in_bin = (qtrajs[i][idxs] > qbins[n]) & (qtrajs[i][idxs] <= qbins[n + 1])
                if np.any(frames_in_bin):
                    Alocal_vs_Q[n] += np.sum(Alocal[frames_in_bin])
                    Anonlocal_vs_Q[n] += np.sum(Anonlocal[frames_in_bin])
                    bin_counts[n] += np.sum(frames_in_bin)
            start_idx += chunk.n_frames

    not0 = np.nonzero(bin_counts)
    Alocal_vs_Q[not0] = (Alocal_vs_Q[not0].T/bin_counts[not0]).T
    Anonlocal_vs_Q[not0] = (Anonlocal_vs_Q[not0].T/bin_counts[not0]).T

    if not os.path.exists("Alocal_vs_Qtanh_0_05"):
        os.mkdir("Alocal_vs_Qtanh_0_05")
    os.chdir("Alocal_vs_Qtanh_0_05")
    np.savetxt("Q_mid_bin.dat", q_mid_bin)
    np.save("Alocal_vs_Q.npy", Alocal_vs_Q)
    np.save("Anonlocal_vs_Q.npy", Anonlocal_vs_Q)
    os.chdir("..")

