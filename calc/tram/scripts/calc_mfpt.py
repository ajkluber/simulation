import os
import argparse
import numpy as np 

import simulation.calc.tram.util as util

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("coordfile")
    parser.add_argument("lagtime", type=int)

    #lagtime = 200
    #coordfile = "Qtanh_0_05.npy"

    args = parser.parse_args()
    lagtime = args.lagtime
    coordfile = args.coordfile

    coordname = coordfile.split(".")[0]

    dirs, dtrajs, lagtimes, models = util.load_markov_state_models()

    obs_trajs = [ np.load(x + "/" + coordfile) for x in dirs ]

    model_msm = models[7]
    clust_idxs = np.arange(model_msm.nstates)

    clust_obs = util.sort_observable_into_clusters(clust_idxs, obs_trajs, dtrajs) 

    obs_avg_clust = np.array([ np.mean(x) for x in clust_obs ])

    # calculate mean first-passage time between minima according to the free
    # energy profile along Q.
    A, B = np.loadtxt(coordname + "_profile/minima.dat")

    left_min = np.argwhere((obs_avg_clust > A*0.95) & (obs_avg_clust < A*1.05))[:,0]
    right_min = np.argwhere((obs_avg_clust > B*0.95) & (obs_avg_clust < B*1.05))[:,0]

    mfpt1 = lagtime*model_msm.mfpt(left_min, right_min)
    mfpt2 = lagtime*model_msm.mfpt(right_min, left_min)

    with open("msm/tfold.dat", "w") as fout:
        fout.write(str(mfpt1))

    with open("msm/tufold.dat", "w") as fout:
        fout.write(str(mfpt2))

    # can compare with the time to go between metastable states
    model_msm.pcca(2)
    left_well = model_msm.metastable_sets[0]
    right_well = model_msm.metastable_sets[1]
    mfpt_pcca1 = lagtime*model_msm.mfpt(left_well, right_well)
    mfpt_pcca2 = lagtime*model_msm.mfpt(right_well, left_well)

    q_avg_meta1 = np.dot(model_msm.stationary_distribution[left_well], obs_avg_clust[left_well])
    q_avg_meta2 = np.dot(model_msm.stationary_distribution[right_well], obs_avg_clust[right_well])

    if q_avg_meta1 < q_avg_meta2:
        with open("msm/tfold_pcca.dat", "w") as fout:
            fout.write(str(mfpt_pcca1))
        with open("msm/tunfold_pcca.dat", "w") as fout:
            fout.write(str(mfpt_pcca2))
    else:
        with open("msm/tfold_pcca.dat", "w") as fout:
            fout.write(str(mfpt_pcca2))
        with open("msm/tunfold_pcca.dat", "w") as fout:
            fout.write(str(mfpt_pcca1))

