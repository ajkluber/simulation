import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
cmap = cm.get_cmap("viridis")

import mdtraj as md
import pyemma.coordinates as coor
import pyemma.msm as msm

import simulation.calc.tram.util as util

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("n_native_pairs", type=int)
    parser.add_argument("pairwise_params_file")

    args = parser.parse_args()
    name = args.name
    n_native_pairs = args.n_native_pairs
    pairwise_file = args.pairwise_params_file

    n_qbins = 40
    topname = "ref.pdb"
    trajname = "traj.xtc"

    with open("Qtanh_0_05_profile/T_used.dat","r") as fin:
        T = float(fin.read())
    tempdirs = [ "T_{:.2f}_{}".format(T, x) for x in [1,2,3] ]

    topfile = tempdirs[0] + "/" + topname
    trajfiles = [ x + "/" + trajname for x in tempdirs ]

    qtrajs = [ np.load(x + "/Qtanh_0_05.npy") for x in tempdirs ]

    #topfile = topname
    #trajfiles = [ trajname ]

    # add features
    non_feat = coor.featurizer(topfile)
    non_feat, feature_info = util.sbm_contact_features(non_feat, pairwise_file, n_native_pairs, skip_nn=1, nonnative_only=True)
    n_res = non_feat.topology.n_residues

    #nat_feat = coor.featurizer(topfile)
    #nat_feat, feature_info = util.sbm_contact_features(nat_feat, pairwise_file, n_native_pairs, skip_nn=1, nonnative_only=True)

    n_non_dim = feature_info['dim']
    non_pairs = feature_info['pairs']

    P_q, qbins = np.histogram(np.concatenate(qtrajs), bins=n_qbins, density=True)
    P_q *= (qbins[1] - qbins[0])
    q_mid_bin = 0.5*(qbins[:-1] + qbins[1:])

    # calculate contacts
    Ai_vs_Q = np.zeros((n_qbins, n_non_dim), float)
    bin_counts = np.zeros(n_qbins)
    for i in range(len(trajfiles)):
        start_idx = 0
        for chunk in md.iterload(trajfiles[i], top=non_feat.topologyfile):
            idxs = np.arange(chunk.n_frames) + start_idx
            Ai_chunk = non_feat.transform(chunk)
            for n in range(n_qbins - 1):
                frames_in_bin = (qtrajs[i][idxs] > qbins[n]) & (qtrajs[i][idxs] <= qbins[n + 1])
                if np.any(frames_in_bin):
                    Ai_vs_Q[n,:] += np.sum(Ai_chunk[frames_in_bin,:], axis=0)
                    bin_counts[n] += np.sum(frames_in_bin)
            start_idx += chunk.n_frames

    not0 = np.nonzero(bin_counts)
    Ai_vs_Q[not0] = (Ai_vs_Q[not0].T/bin_counts[not0]).T

    if not os.path.exists("Ai_vs_Qtanh_0_05"):
        os.mkdir("Ai_vs_Qtanh_0_05")
    os.chdir("Ai_vs_Qtanh_0_05")


    #C_vs_Q = np.zeros((n_qbins, n_res, n_res))
    A_vs_loop_vs_Q = np.zeros((n_qbins, n_res - 3))
    color_idxs = np.arange(n_qbins)/float(n_qbins)
    for n in range(n_qbins):
        # contact map
        C = np.nan*np.zeros((n_res, n_res))
        for i in range(n_non_dim):
            C[non_pairs[i,1], non_pairs[i,0]] = Ai_vs_Q[n, i]
        C_ma = np.ma.array(C, mask=np.isnan(C))

        # average non-native contacts versus loop length
        A_vs_loop = np.array([ np.ma.mean(C_ma.diagonal(-i)) for i in range(3, n_res) ])
        loop_length = np.arange(3, n_res)
        A_vs_loop_vs_Q[n] = A_vs_loop

        log_C_ma = np.ma.log(C_ma)
            
        # plot contact map
        #plt.figure(n + 2)
        #plt.pcolormesh(log_C_ma, vmin=-9, vmax=0, cmap='viridis')
        #plt.xlabel("residue i")
        #plt.xlabel("residue j")
        #plt.title("Contact frequency $Q = {}$".format(q_mid_bin[9]))
        #cbar = plt.colorbar()
        #cbar.set_label("$\\log Q_{ij}$")
        ##plt.savefig("")

        #plt.figure()
        plt.plot(loop_length, A_vs_loop, color=cmap(color_idxs[n]))
        #plt.semilogy()
        plt.xlabel("Loop length")
        #plt.ylabel("$\\log A$")
        plt.ylabel("Average non-native contacts")


    A_vs_loop_vs_Q_ma = np.ma.array(A_vs_loop_vs_Q, mask=np.isnan(A_vs_loop_vs_Q))

    plt.figure()
    plt.pcolormesh(A_vs_loop_vs_Q_ma.T, cmap="viridis")
    plt.xlabel("Q")
    plt.ylabel("Loop length")
    plt.colorbar()

    plt.show()

    # save
    #np.savetxt("nonnative_pairs.dat", non_pairs, fmt="%4d") 
    #np.savetxt("loop_length.dat", loop_length, fmt="%4d")
    #np.savetxt("A_vs_loop.dat", A_vs_loop)

    os.chdir("..")
