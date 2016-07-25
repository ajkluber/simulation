import os
import pickle
import argparse
import numpy as np

import mdtraj as md
import pyemma.coordinates as coor
import pyemma.msm as msm

import util

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pairwise_file")
    parser.add_argument("n_native_pairs", type=int)
    parser.add_argument("tica_lag", type=int)
    parser.add_argument("--trajname", default="traj.xtc", type=str)
    parser.add_argument("--topname", default="ref.pdb", type=str)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--n_clusters", default=400, type=int)
    parser.add_argument("--tica_dims", default=10, type=int)
    parser.add_argument("--recluster", action="store_true")
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    pairwise_file = args.pairwise_file
    n_native_pairs = args.n_native_pairs
    tica_lag = args.tica_lag
    trajname = args.trajname
    topname = args.topname
    stride = args.stride
    n_clusters = args.n_clusters    
    tica_dims = args.tica_dims    
    recluster = args.recluster    
    display = args.display    

    with open("Qtanh_0_05_profile/T_used.dat","r") as fin: 
        T = float(fin.read())

    tempdirs = [ "T_{:.2f}_{}".format(T, x) for x in [1,2,3] ]

    topfile = tempdirs[0] + "/" + topname

    trajfiles = [ x + "/" + trajname for x in tempdirs ]

    # add features.
    feat = coor.featurizer(topfile)
    feat = util.sbm_contact_features(feat, pairwise_file, n_native_pairs)

    if not os.path.exists("msm"):
        os.mkdir("msm")

    if (not os.path.exists("msm/dtrajs.pkl")) or recluster:
        # cluster if necessary
        inp = coor.source(trajfiles, feat)
        tica_obj = coor.tica(inp, dim=tica_dims, lag=tica_lag, kinetic_map=True)
        Y = tica_obj.get_output()
        cl = coor.cluster_kmeans(data=Y, k=n_clusters)
        dtrajs = cl.dtrajs

        os.chdir("msm")
        dirs = [ os.path.basename(os.path.dirname(x)) for x in trajfiles ]
        dtraj_info = { dirs[x]:dtrajs[x] for x in range(len(dirs)) }
        dtraj_info["dirs"] = dirs
        with open("dtrajs.pkl", 'wb') as fhandle:
            pickle.dump(dtraj_info, fhandle)
    else:
        os.chdir("msm")
        with open("dtrajs.pkl", 'rb') as fhandle:
            dtraj_pkl = pickle.load(fhandle)
            dirs = dtraj_pkl["dirs"]
            dtrajs = [ dtraj_pkl[x] for x in dirs ]

    # estimate MSM's at different lagtimes
    lags = [1,2,5,10,20,50,100,200,300,400,500,600,700,800,900,1000]
    its = msm.its(dtrajs, lags=lags)
    util.save_markov_state_models(T, its.models)

    # save MSM's 
    if display:
        import pyemma.plots as mplt
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pyemma.plots as mplt

    mplt.plot_implied_timescales(its, ylog=False)
    plt.title("T = " + str(T))
    plt.savefig("its_tanh_cont_features.pdf")
    plt.savefig("its_tanh_cont_features.png")
    os.chdir("..")
    
    if display:
        plt.show()

