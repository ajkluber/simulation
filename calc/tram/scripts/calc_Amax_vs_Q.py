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

    P_q, qbins = np.histogram(np.concatenate(qtrajs), bins=n_qbins, density=True)
    P_q *= (qbins[1] - qbins[0])
    q_mid_bin = 0.5*(qbins[:-1] + qbins[1:])

    # calculate contacts as a function of Q.
    Amax_vs_Q = np.zeros(n_qbins, float)
    for i in range(len(trajfiles)):
        start_idx = 0
        for chunk in md.iterload(trajfiles[i], top=non_feat.topologyfile):
            idxs = np.arange(chunk.n_frames) + start_idx
            A = np.sum(non_feat.transform(chunk), axis=1)

            for n in range(n_qbins - 1):
                frames_in_bin = (qtrajs[i][idxs] > qbins[n]) & (qtrajs[i][idxs] <= qbins[n + 1])
                if np.any(frames_in_bin):
                    Amax_vs_Q[n] = np.max([ Amax_vs_Q[n], np.max(A[frames_in_bin]) ])
            start_idx += chunk.n_frames

    if not os.path.exists("Amax_vs_Qtanh_0_05"):
        os.mkdir("Amax_vs_Qtanh_0_05")
    os.chdir("Amax_vs_Qtanh_0_05")
    np.savetxt("Q_mid_bin.dat", q_mid_bin)
    np.save("Amax_vs_Q.npy", Amax_vs_Q)
    os.chdir("..")

