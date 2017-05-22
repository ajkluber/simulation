import os
import glob
import numpy as np

import mdtraj as md

if __name__ == "__main__":
    # get directories
    topname = "ref.pdb"
    trajname = "traj.xtc"
    with open("Qtanh_0_05_profile/T_used.dat", "r") as fin:
        T = float(fin.read())

    tempdirs = [ "T_{:.2f}_{}".format(T, x) for x in [1,2,3] ]
    topfile = tempdirs[0] + "/" + topname

    pdb = md.load(topfile)
    unfolded_U = np.loadtxt("Qtanh_0_05_profile/minima.dat")[0]

    corr_ij = np.zeros(pdb.n_residues - 2)
    total_frames = 0.
    dot_ij = np.nan*np.zeros((pdb.n_residues - 1, pdb.n_residues - 1))

    for n in range(len(tempdirs)):
        traj = md.load(tempdirs[n] + "/" + trajname, top=topfile)

        # grab vectors in the unfolded state
        q = np.load(tempdirs[n] + "/Qtanh_0_05.npy")
        U = (q <= unfolded_U)

        # tangent vectors along chain
        r_i = traj.xyz[U,1:,:] - traj.xyz[U,:-1,:]
        v_i = np.array([ (r_i[:,x,:].T/np.linalg.norm(r_i[:,x,:], axis=1)).T for x in range(traj.n_residues - 1) ])
        
        # compute correlation along the chain
        for i in range(1, traj.n_residues - 2):
            for j in range(i, traj.n_residues - 1):
                if n == 0:
                    dot_ij[i,j] = np.sum(v_i[j,:,:]*v_i[i,:,:])
                else:
                    dot_ij[i,j] += np.sum(v_i[j,:,:]*v_i[i,:,:])
        total_frames += float(np.sum(U))

    dot_ij_ma = np.ma.array(dot_ij, mask=np.isnan(dot_ij))/total_frames

    for i in range(traj.n_residues - 2):
        corr_ij[i] += np.ma.mean(np.ma.diagonal(dot_ij_ma, offset=i))

    if not os.path.exists("persist_length"):
        os.mkdir("persist_length")
    np.save("persist_length/corr_ij.npy", corr_ij)

