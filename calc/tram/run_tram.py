import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

import mdtraj as md
import pyemma
import pyemma.coordinates as coor
import pyemma.msm as msm
import pyemma.plots as mplt

import util 
global KB
KB = 0.0083145

#    # estimate MSM at one temperature to get idea of lagtime
#    lags = [1,2,5,10,20,50,100,400,1000]
#    its = msm.its(dtrajs, lags=lags)
#
#    mplt.plot_implied_timescales(its, ylog=False)
#    plt.title("msm/{}".format(name))
#    plt.savefig("msm/{}_its.png".format(name),bbox_inches="tight")
#    plt.savefig("msm/{}_its.pdf".format(name),bbox_inches="tight")

class DummyTram(object):
    def __init__(self):
        pass


def assign_pair_distance_dihedral_features(topfile, pairsfile):
    """Assign features"""
    ref = md.load(topfile)
    dihedral_idxs = np.array([ [i, i + 1, i + 2, i + 3] for i in range(ref.n_residues - 3) ])
    pair_idxs = np.loadtxt(pairsfile, dtype=int) - 1

    # cluster trajectories
    feat = coor.featurizer(topfile)
    feat.add_distances(pair_idxs)
    feat.add_dihedrals(dihedral_idxs)

    return feat

def cluster_and_estimate_dtram(feat, trajfiles, T, stride=1, tica_lag=100, tram_lag=400):

    dirs = [ os.path.dirname(x) for x in trajfiles ]
    beta = [ 1./(KB*x) for x in T ]

    inp = coor.source(trajfiles, feat)

    tica_obj = coor.tica(inp, lag=100, kinetic_map=True, stride=stride)
    Y = tica_obj.get_output()

    cl = coor.cluster_kmeans(data=Y, stride=stride)
    dtrajs = cl.dtrajs

    # dimensionless energy
    energy_trajs = [ beta[i]*np.loadtxt("{}/Etot.dat".format(dirs[i]), usecols=(1,)) for i in range(len(dirs)) ]
    temp_trajs = [ KB*T[i]*np.ones(energy_trajs[i].shape[0], float) for i in range(len(dirs)) ]


    # TRAM approach
    tram = pyemma.thermo.estimate_multi_temperature(energy_trajs, temp_trajs,
            dtrajs, energy_unit='kT', temp_unit='kT', estimator='dtram',
            lag=tram_lag)

    return dirs, dtrajs, tram

def save_tram_data(dirs, dtrajs, tram):
    # save information needed to rebuild the MSM's created by dTRAM.
    if not os.path.exists("dtram"):
        os.mkdir("dtram")
    os.chdir("dtram")

    dtraj_info = { dirs[x]:dtrajs[x] for x in range(len(dirs)) }
    dtraj_info["dirs"] = dirs
    with open("dtrajs.pkl", "wb") as fhandle:
        pickle.dump(dtraj_info, fhandle)

    tram_info = {}
    tram_info["temperatures"] = tram.temperatures/KB
    tram_info["tram_f"] = tram.f
    tram_info["tram_f_therm"] = tram.f_therm

    for k in range(tram.nthermo):
        temperature = tram.temperatures[k]/KB
        tram_info["{:.2f}_active_set".format(temperature)] = tram.model_active_set[k]
        tram_info["{:.2f}_stationary_distribution".format(temperature)] = tram.models[k].stationary_distribution
        tram_info["{:.2f}_transition_matrix".format(temperature)] = tram.models[k].transition_matrix

    with open("dtram.pkl", "wb") as fhandle:
        pickle.dump(tram_info, fhandle)

    os.chdir("..")

def load_dtram():

    os.chdir("dtram")
    with open("dtrajs.pkl", "rb") as fhandle:
        dtraj_pkl = pickle.load(fhandle)
        dirs = dtraj_pkl["dirs"]
        dtrajs = [ dtraj_pkl[x] for x in dirs ]

    dtram = DummyTram()
    with open("dtram.pkl", "rb") as fhandle:
        dtram_pkl = pickle.load(fhandle)
        dtram.temperatures = dtram_pkl["temperatures"]*KB
        dtram.f = dtram_pkl["tram_f"]
        dtram.f_therm = dtram_pkl["tram_f_therm"]
        dtram.ntherm = len(dtram.temperatures)

        dtram.model_active_set = [ dtram_pkl["{:.2f}_active_set".format(x)] for x in dtram.temperatures/KB ]
        model_pi = [ dtram_pkl["{:.2f}_stationary_distribution".format(x)] for x in dtram.temperatures/KB ]
        model_T = [ dtram_pkl["{:.2f}_transition_matrix".format(x)] for x in dtram.temperatures/KB ]
        dtram.models = [ msm.MSM(model_T[x], pi=model_pi[x]) for x in range(dtram.ntherm) ]

    os.chdir("..")
    return dirs, dtrajs, dtram

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
        feat = assign_pair_distance_dihedral_features(topfile, pairsfile)
        dirs, dtrajs, dtram = cluster_and_estimate_dtram(feat, trajfiles, T, tram_lag=tram_lag)
        save_tram_data(dirs, dtrajs, dtram)
    else:
        print "loading tram"
        dirs, dtrajs, dtram = load_dtram()
    
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
