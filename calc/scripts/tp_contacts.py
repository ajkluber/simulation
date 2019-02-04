import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mdtraj as md

import simulation.calc.transits as transits
import simulation.calc.observables as observables

def get_n_native_pairs(name):
    if os.path.exists(name + ".contacts"):
        n_native_pairs = len(np.loadtxt(name + ".contacts"))
    elif os.path.exists(name + ".ini"):
        with open(name + ".ini", "r") as fin:
            n_native_pairs = int([ x for x in fin.readlines() if x.startswith("n_native_pairs") ][0].split()[-1])
    else:
        return False
    return n_native_pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name",
            type=str,
            help="Name of protein.")

    args = parser.parse_args()
    name = args.name

    coordfile = "Qtanh_0_05.npy"
    coordname = coordfile.split(".")[0]

    n_native_pairs = get_n_native_pairs(name)
    with open("{}_profile/T_used.dat".format(coordname), "r") as fin:
        T_used = float(fin.read())

    minima = np.loadtxt("{}_profile/minima.dat".format(coordname))/float(n_native_pairs)
    U, N = minima

    qtrajs = [ np.load("T_{:.2f}_{}/{}".format(T_used, n, coordfile))/float(n_native_pairs) for n in [1,2,3] ]
    trajfiles = [ "T_{:.2f}_{}/traj.xtc".format(T_used, n) for n in [1,2,3] ]
    topfile = "T_{:.2f}_1/ref.pdb".format(T_used)

    # make a contact observable
    pairs = np.loadtxt(name + "_pairwise_params", usecols=(0,1), dtype=int) - 1
    n_pairs = len(pairs)

    r0 = np.loadtxt(name + "_pairwise_params", usecols=(5,), dtype=float)
    r0_cont = r0 + 0.1
    widths = 0.05*np.ones(n_pairs)
    q_obs = observables.TanhContacts(topfile, pairs, r0_cont, widths, periodic=False)

    # bin contact maps along transition paths
    bin_edges = np.linspace(U + (5./n_native_pairs), N - (5./n_native_pairs), 6)
    bin_counts = np.zeros(len(bin_edges) - 1, float)
    bin_sum = np.zeros((len(bin_edges) - 1, n_pairs), float)
    for i in range(len(trajfiles)):
        print "traj ", i
        q = qtrajs[i]
        trajfile = trajfiles[i]

        dtraj = np.zeros(len(q))
        dtraj[q <= U] = 0
        dtraj[q >= N] = 2
        dtraj[(q > U) & (q < N)] = 1
        dwells1, dwells2, transits12, transits21 = transits.partition_dtraj(dtraj, 0, 2)

        traj = md.load(trajfile, top=topfile)
        for t in range(len(transits12)):
            # calculate the contacts on each transition path
            beg = transits12[t][0]
            end = beg + transits12[t][1]
            q_tp = q[beg:end]
            traj_tp = traj[beg:end]
            conts = np.array(q_obs.map(traj_tp))
            #plt.plot(q[beg:end])
            Q = np.sum(conts[:,:n_native_pairs], axis=0)
            for n in range(len(bin_edges) - 1): 
                frames_in_this_bin = (q_tp > bin_edges[n]) & (q_tp <= bin_edges[n+1])
                if np.any(frames_in_this_bin):
                    bin_sum[n,:] += np.sum(conts[frames_in_this_bin,:], axis=0)
                    bin_counts[n] += np.sum(frames_in_this_bin)
            
    bin_avg = np.zeros((len(bin_sum), n_pairs), float)
    for n in range(len(bin_edges) - 1):
        if bin_counts[n] > 0:
            bin_avg[n] = bin_sum[n,:]/bin_counts[n]

    if not os.path.exists("{}_tp_contacts".format(coordname)):
        os.mkdir("{}_tp_contacts".format(coordname))
    os.chdir("{}_tp_contacts".format(coordname))
    np.save("bin_avg.npy", bin_avg)
    np.save("bin_edges.npy", bin_edges)
    os.chdir("..")

    mid_bin = 0.5*(bin_edges[:-1] + bin_edges[1:])
    fig, axes = plt.subplots(1, 5, figsize=(15,4))
    for n in range(len(bin_edges) - 1):
        C = np.zeros((traj.n_residues, traj.n_residues), float)
        for i in range(n_pairs):
            if i < n_native_pairs:
                C[pairs[i,1], pairs[i,0]] = bin_avg[n,i]
            else:
                C[pairs[i,1], pairs[i,0]] = -bin_avg[n,i]

        ax = axes[n]
        ax.pcolormesh(C, vmin=-1, vmax=1, cmap="bwr_r")
        ax.set_title(r"$Q = {:.2f}$".format(mid_bin[n]), fontsize=14)
        ax.set_xlim(0, traj.n_residues)
        ax.set_ylim(0, traj.n_residues)
        ax.set(adjustable='box-forced', aspect='equal')
    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.05)
    fig.suptitle(name + " transition path contacts", fontsize=18)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig("plots/{}_tp_contacts.pdf".format(coordname))
    plt.savefig("plots/{}_tp_contacts.png".format(coordname))
