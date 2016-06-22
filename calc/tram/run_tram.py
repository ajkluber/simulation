import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

import mdtraj as md
import pyemma.thermo as thermo
import pyemma.coordinates as coor
import pyemma.msm as msm
import pyemma.plots as mplt

import util 

global KB
KB = 0.0083145

class DummyTram(object):
    def __init__(self):
        pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pairsfile")
    parser.add_argument("tempsfile")
    parser.add_argument("tica_lag", type=int)
    parser.add_argument("tram_lag", type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    print args

    pairsfile = args.pairsfile
    tempsfile = args.tempsfile
    tica_lag = args.tica_lag
    tram_lag = args.tram_lag
    stride = args.stride
    display = args.display    

    #name = "SH3"
    #pairsfile = "{}.contacts".format(name)
    #stride = 1
    #tempsfile = "temp_dirs"
    cwd = os.getcwd()

    organized_temps = util.get_organized_temps(tempsfile)
    sorted_temps = organized_temps.keys()
    sorted_temps.sort()
    
    dirs = [ dir for key in sorted_temps for dir in organized_temps[key] ]

    topfile = "{}/ref.pdb".format(dirs[0])
    trajfiles = [ "{}/traj.xtc".format(x) for x in dirs ]

    T_labels = [ x.split("_")[1] for x in dirs ]
    T = [ float(x) for x in T_labels ]

    #tram_lag = 400  # Found from doing an MSM at one temp. For C-alpha SBM

    if not os.path.exists("dtram/dtram.pkl"):
        # estimate dtram 
        print "solving tram"
        feat = coor.featurizer(topfile)
        feat = util.default_ca_sbm_features(feat, topfile, pairsfile=pairsfile)
        dirs, dtrajs, dtram = util.multi_temperature_dtram(feat, trajfiles, T, tram_lag=tram_lag)
        util.save_multi_temperature_dtram(dirs, dtrajs, dtram)
    else:
        print "loading tram"
        dirs, dtrajs, dtram = util.load_multi_temperature_dtram()
    
    # define bin edges for clustering observable
    bins = np.linspace(0, 133, 50)
    mid_bin = 0.5*(bins[1:] + bins[:-1])

    # calculate the distribution of an observable from the tram MSM's.

    # get observables for each cluster at each thermodynamic state
    thermo_obs = {}
    for k in range(dtram.nthermo):
        Tkey = sorted_temps[k]
        
        # idxs of occupied clusters at this thermo state
        clust_idxs = dtram.model_active_set[k]
        clust_obs = [ None for x in range(len(clust_idxs)) ]

        # get data from all directories at this thermo state.
        for Tdir in organized_temps[Tkey]:
            print "sorting frames from: ", Tdir
            os.chdir(Tdir)
            dir_idx = dirs.index(Tdir) 
            
            q = np.loadtxt("qtanh.dat") # todo: generalize observable
            dtraj = dtrajs[dir_idx]
            for i in range(len(clust_idxs)):
                # collect the frames that are in each cluster
                obs_in_clust = (dtraj == clust_idxs[i])
                if np.any(obs_in_clust):
                    if clust_obs[i] is None:
                        clust_obs[i] = q[obs_in_clust]
                    else:
                        clust_obs[i] = np.concatenate((clust_obs[i], q[obs_in_clust]))

            os.chdir(cwd)
        thermo_obs[Tkey] = clust_obs

    # plot the overall observable distribution
    thermo_dist = {}
    plt.figure()
    for k in range(dtram.nthermo):
        Tkey = sorted_temps[k]
        clust_idxs = dtram.model_active_set[k]
        clust_obs = thermo_obs[Tkey]  

        # combine cluster distributions with weights from stationary
        # distribution.
        pi = dtram.models[k].stationary_distribution
        P_A = np.zeros(len(bins) - 1)
        for i in range(len(clust_idxs)):
            P_A_i, bins = np.histogram(clust_obs[i], bins=bins, density=True)
            P_A += pi[i]*P_A_i
        thermo_dist[Tkey] = P_A

        pmf = -np.log(P_A)
        pmf -= pmf.min()
        plt.plot(mid_bin, pmf, label="{}".format(Tkey))
        #plt.plot(mid_bin, P_A, label="{}".format(Tkey))

    plt.legend()
    plt.title("distribution $Q_{tanh}$")
    plt.xlabel("$Q_{tanh}$")
    plt.ylabel("distribution")
    plt.show()
