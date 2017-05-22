import os
import argparse
import numpy as np 

import pyemma.coordinates as coor
import pyemma.util.discrete_trajectories as dt

import simulation.calc.tram.util as util

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajname", default="traj.xtc", type=str)
    parser.add_argument("--topname", default="ref.pdb", type=str)

    args = parser.parse_args()
    trajname = args.trajname
    topname = args.topname

    lag_idx = 7
    n_sample = 100

    with open("Qtanh_0_05_profile/T_used.dat","r") as fin: 
        T = float(fin.read())

    tempdirs = [ "T_{:.2f}_{}".format(T, x) for x in [1,2,3] ]
    topfile = tempdirs[0] + "/" + topname
    trajfiles = [ x + "/" + trajname for x in tempdirs ]

    print "initializing featurizer"
    # initialize traj input info.
    feat = coor.featurizer(topfile)
    inp = coor.source(trajfiles, feat)

    print "loading MSMs"
    # Load MSM's that have already been calculated.
    dirs, dtrajs, lagtimes, models = util.load_markov_state_models()

    # lagtime of 200
    tau = lagtimes[lag_idx]
    model_msm = models[lag_idx] 

    with open("msm/n_dominant_timescales.dat", "r") as fin:
        n_pcca = int(fin.read()) + 1

    print "grabbing frames"
    # Grab frames from pcca clustering
    model_msm.pcca(n_pcca)
    pcca_dist = model_msm.metastable_distributions
    active_state_indexes = dt.index_states(dtrajs)
    pcca_samples = dt.sample_indexes_by_distribution(active_state_indexes, pcca_dist, n_sample)

    if not os.path.exists("msm/PCCA_N_" + str(n_pcca)):
        os.mkdir("msm/PCCA_N_" + str(n_pcca))

    outfiles = [ 'msm/PCCA_N_{}/pcca{}.xtc'.format(n_pcca, x) for x in range(1, n_pcca + 1) ]
                    
    print "saving"
    coor.save_trajs(inp, pcca_samples, outfiles=outfiles)

