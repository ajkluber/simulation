import os
import argparse
import numpy as np

import mdtraj as md

# SCAN MANY DISTANCES 
#    count = 1
#    idxs = range(0, 70, 4)
#    #for start_idx in idxs:
#    for n in range(8, len(idxs)):
#        start_idx = idxs[n]
#        print "pair ({}/{})".format(n, len(idxs))
#        count += 1
#
#        pair_idxs = np.array([[0, x] for x in range(86) if np.abs(start_idx - x) > 8])
#
#        os.chdir("T_{:.2f}_1".format(T_used))
#        traj = md.load("traj.xtc", top="ref.pdb")
#        ext_r = md.compute_distances(traj, pair_idxs)
#        os.chdir("..")
#
#        bins = np.linspace(0, 6.2, 500) 
#
#        n_dim = int(np.ceil(np.sqrt(float(len(pair_idxs)))))
#        fig, axes = plt.subplots(n_dim, n_dim, figsize=(10,10), sharex=True)
#        for i in range(len(pair_idxs)):
#            ax = axes[i / n_dim, i % n_dim]
#            ax.hist(ext_r[:,i], bins=bins, histtype="stepfilled", normed=True)
#            ax.set_xticks([])
#            ax.set_yticks([])
#            #ax.annotate
#        plt.subplots_adjust(wspace=0, hspace=0)
#        plt.show()

if __name__ == "__main__":

    with open("Qtanh_0_05_profile/T_used.dat", "r") as fin:
        T_used = float(fin.read())

    pdb = md.load("T_{:.2f}_1/ref.pdb".format(T_used))
    pair_idxs = np.array([[0, pdb.n_residues - 1]]) 

    Tdirs = [ "T_{:.2f}_{}".format(T_used, x) for x in [1,2,3] ]
    trajfile = "traj.xtc"

    for i in range(len(Tdirs)):
        os.chdir(Tdirs[i])
        traj = md.load(trajfile, top=pdb, atom_indices=[0, pdb.n_residues - 1])
        r = md.compute_distances(traj, np.array([[0, 1]]))[:,0]
        #r = []
        #for chunk in md.iterload(trajfile, top=pdb, atom_indices=[0, pdb.n_residues - 1]):
        #    #r.append(md.compute_distances(chunk, pair_idxs)[:,0])
        #    r.append(md.compute_distances(chunk, np.array([[0, 1]]))[:,0])
        #r = np.concatenate(r)
        #np.save("ext_{}_{}.npy".format(pair_idxs[0,0], pair_idxs[0,1]), r)
        np.save("r1N.npy", r)
        os.chdir("..")

