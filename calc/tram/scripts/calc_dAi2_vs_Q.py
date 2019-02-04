import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

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
    cont_feat = coor.featurizer(topfile)
    cont_feat, feature_info = util.sbm_contact_features(cont_feat, pairwise_file, n_native_pairs, skip_nn=1)
    n_res = cont_feat.topology.n_residues

    pairs = feature_info['pairs']
    non_pairs = pairs[n_native_pairs:,:]
    nat_pairs = pairs[:n_native_pairs,:]
    n_non_pairs = len(non_pairs)

    P_q, qbins = np.histogram(np.concatenate(qtrajs), bins=n_qbins, density=True)
    P_q *= (qbins[1] - qbins[0])
    q_mid_bin = 0.5*(qbins[:-1] + qbins[1:])

    Ai_vs_Q = np.load("Ai_vs_Qtanh_0_05/Ai_vs_Q.npy")
    Qi_vs_Q = np.load("Qi_vs_Qtanh_0_05/Qi_vs_Q.npy")

    # calculate variance as a function of Q.
    Ai_var_vs_Q = np.zeros((n_qbins, n_non_pairs), float)
    Qi_var_vs_Q = np.zeros((n_qbins, n_native_pairs), float)
    bin_counts = np.zeros(n_qbins)
    for i in range(len(trajfiles)):
        start_idx = 0
        for chunk in md.iterload(trajfiles[i], top=cont_feat.topologyfile):
            idxs = np.arange(chunk.n_frames) + start_idx
            obs_chunk = cont_feat.transform(chunk)
            Ai_chunk = obs_chunk[:,n_native_pairs:]
            Qi_chunk = obs_chunk[:,:n_native_pairs]
            for n in range(n_qbins - 1):
                frames_in_bin = (qtrajs[i][idxs] > qbins[n]) & (qtrajs[i][idxs] <= qbins[n + 1])
                if np.any(frames_in_bin):
                    Ai_var_vs_Q[n,:] += np.sum((Ai_chunk[frames_in_bin,:] - Ai_vs_Q[n,:])**2, axis=0)
                    Qi_var_vs_Q[n,:] += np.sum((Qi_chunk[frames_in_bin,:] - Qi_vs_Q[n,:])**2, axis=0)
                    bin_counts[n] += np.sum(frames_in_bin)
            start_idx += chunk.n_frames

    not0 = np.nonzero(bin_counts)
    Ai_var_vs_Q[not0] = (Ai_var_vs_Q[not0].T/bin_counts[not0]).T
    Qi_var_vs_Q[not0] = (Qi_var_vs_Q[not0].T/bin_counts[not0]).T

#    C = np.zeros((n_res, n_res), float)
#    for i in range(n_non_pairs):
#        C[non_pairs[i,1], non_pairs[i,0]] = -Ai_var_vs_Q[12,i] 
#
#    for i in range(n_native_pairs):
#        C[nat_pairs[i,1], nat_pairs[i,0]] = Qi_var_vs_Q[12,i] 
#
#    #plt.pcolormesh(C, cmap="bwr_r", vmin=-0.08, vmax=0.08)
#    plt.pcolormesh(C, cmap="bwr_r", vmin=-0.2, vmax=0.2)
#    plt.colorbar()
#    plt.show()

    np.save("Ai_vs_Qtanh_0_05/Ai_var_vs_Q.npy", Ai_var_vs_Q)
    np.save("Qi_vs_Qtanh_0_05/Qi_var_vs_Q.npy", Qi_var_vs_Q)

