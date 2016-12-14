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

    with open("Qtanh_0_05_profile/T_used.dat","r") as fin: 
        T = float(fin.read())

    tempdirs = [ "T_{:.2f}_{}".format(T, x) for x in [1,2,3] ]
    topfile = tempdirs[0] + "/" + topname
    trajfiles = [ x + "/" + trajname for x in tempdirs ]

    # initialize traj input info.
    feat = coor.featurizer(topfile)
    inp = coor.source(trajfiles, feat)

    # Load MSM's that have already been calculated.
    dirs, dtrajs, lagtimes, models = util.load_markov_state_models()

    model_msm = models[7] # lagtime of 200

    # Determine the number of clusters by the number of timescales.
    n_pcca = 2
    n_sample = 100

    # Grab frames from pcca clustering
    model_msm.pcca(n_pcca)
    pcca_dist = model_msm.metastable_distributions
    active_state_indexes = dt.index_states(dtrajs)
    pcca_samples = dt.sample_indexes_by_distribution(active_state_indexes, pcca_dist, n_sample)

    outfiles = [ 'msm/pcca{}.xtc'.format(x) for x in range(1, n_pcca + 1) ]
                    
    coor.save_trajs(inp, pcca_samples, outfiles=outfiles)

