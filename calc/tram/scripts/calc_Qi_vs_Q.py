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

    # only calculate native contacts
    # add features
    nat_feat = coor.featurizer(topfile)
    nat_feat, feature_info = util.sbm_contact_features(nat_feat, pairwise_file, n_native_pairs, skip_nn=1, native_only=True)
    n_res = nat_feat.topology.n_residues

    nat_pairs = feature_info['pairs']

    P_q, qbins = np.histogram(np.concatenate(qtrajs), bins=n_qbins, density=True)
    P_q *= (qbins[1] - qbins[0])
    q_mid_bin = 0.5*(qbins[:-1] + qbins[1:])

    # calculate contacts as a function of Q.
    Qi_vs_Q = np.zeros((n_qbins, n_native_pairs), float)
    bin_counts = np.zeros(n_qbins)
    for i in range(len(trajfiles)):
        start_idx = 0
        for chunk in md.iterload(trajfiles[i], top=nat_feat.topologyfile):
            idxs = np.arange(chunk.n_frames) + start_idx
            Qi_chunk = nat_feat.transform(chunk)
            for n in range(n_qbins - 1):
                frames_in_bin = (qtrajs[i][idxs] > qbins[n]) & (qtrajs[i][idxs] <= qbins[n + 1])
                if np.any(frames_in_bin):
                    Qi_vs_Q[n,:] += np.sum(Qi_chunk[frames_in_bin,:], axis=0)
                    bin_counts[n] += np.sum(frames_in_bin)
            start_idx += chunk.n_frames

    not0 = np.nonzero(bin_counts)
    Qi_vs_Q[not0] = (Qi_vs_Q[not0].T/bin_counts[not0]).T


    if not os.path.exists("Qi_vs_Qtanh_0_05"):
        os.mkdir("Qi_vs_Qtanh_0_05")
    os.chdir("Qi_vs_Qtanh_0_05")

    # save
    np.savetxt("Q_mid_bin.dat", q_mid_bin)
    np.savetxt("native_pairs.dat", nat_pairs, fmt="%4d") 
    np.save("Qi_vs_Q.npy", Qi_vs_Q)

    os.chdir("..")

