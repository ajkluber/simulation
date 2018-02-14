import os
import argparse
import numpy as np 

import simulation.calc.tram.util as util

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("coordfile")
    parser.add_argument("lagtime", type=float)
    parser.add_argument("--threshold", type=float, default=0.9)

    #lagtime = 200
    #coordfile = "Qtanh_0_05.npy"

    import time
    starttime = time.time()

    args = parser.parse_args()
    coordfile = args.coordfile
    lagtime = args.lagtime
    threshold = args.threshold

    coordname = coordfile.split(".")[0]

    print "loading MSMs..."
    dirs, dtrajs, lagtimes, models = util.load_markov_state_models()

    obs_trajs = [ np.load(x + "/" + coordfile) for x in dirs ]

    model_msm = models[7]
    clust_idxs = np.arange(model_msm.nstates)

    print "calculating q on clusters..."
    clust_obs = util.sort_observable_into_clusters(clust_idxs, obs_trajs, dtrajs) 
    obs_avg_clust = np.array([ np.mean(x) for x in clust_obs ])

    # calculate mean first-passage time between minima according to the free
    # energy profile along Q. This is a crude (heuristic) way to define sets.
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
    left_well = np.argwhere(model_msm.metastable_memberships[:,0] > threshold)[:,0]
    right_well = np.argwhere(model_msm.metastable_memberships[:,1] > threshold)[:,0]

    #left_well = model_msm.metastable_sets[0]
    #right_well = model_msm.metastable_sets[1]
    mfpt_pcca1 = lagtime*model_msm.mfpt(left_well, right_well)
    mfpt_pcca2 = lagtime*model_msm.mfpt(right_well, left_well)

    #ReactiveFlux = msm.tpt(model_msm, left_well, right_well)

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

    endtime = time.time()
    print "took: {:.4f} min".format((endtime - starttime)/60.)
