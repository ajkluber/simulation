import os
import argparse
import numpy as np 

import simulation.calc.tram.util as util

if __name__ == "__main__":
    """Calculate the correlation coefficient of a reaction coordinate (e.x. Q,
    fraction of native contacts) with MSM eigenvectors which represent the
    dominant relaxation processes of the system.

    For a funneled landscape we would expect the slowest process to correlate
    with Q and that is what we see.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("coordfile")
    parser.add_argument("lagtime", type=int)
    parser.add_argument("--threshold", type=float, default=0.9)

    #lagtime = 200
    #coordfile = "Qtanh_0_05.npy"

    args = parser.parse_args()
    lagtime = args.lagtime
    coordfile = args.coordfile
    threshold = args.threshold

    coordname = coordfile.split(".")[0]

    dirs, dtrajs, lagtimes, models = util.load_markov_state_models()

    obs_trajs = [ np.load(x + "/" + coordfile) for x in dirs ]

    model_msm = models[7]

    # find maximum timescale separation
    ti = lagtime*model_msm.timescales()
    separation = (ti[:15] - ti[1:16])/ti[1:16]
    max_sep = np.argmax(separation) 

    clust_idxs = np.arange(model_msm.nstates)
    clust_obs = util.sort_observable_into_clusters(clust_idxs, obs_trajs, dtrajs) 
    obs_avg_clust = np.array([ np.mean(x) for x in clust_obs ])

    # how does Q correlate with the slowest eigenvector?
    psi1 = model_msm.eigenvectors_right()[:,1]
    pearson_corr = np.mean((obs_avg_clust - obs_avg_clust.mean())*(psi1 - psi1.mean())/(np.std(obs_avg_clust)*np.std(psi1)))

    # how does Q correlate with the slowest eigenvector *in the transition
    # state region* (i.e. outside of the metastable states)?
    model_msm.pcca(2)
    TS_idxs = np.argwhere((model_msm.metastable_memberships[:,0] < threshold) & (model_msm.metastable_memberships[:,1] < threshold))[:,0]
    q_TS = obs_avg_clust[TS_idxs]
    psi1_TS = psi1[TS_idxs]
    pearson_corr_TS = np.mean((q_TS - q_TS.mean())*(psi1_TS - psi1_TS.mean())/(np.std(q_TS)*np.std(psi1_TS)))

    with open("msm/q_vs_psi1_corr.dat", "w") as fout:
        fout.write(str(pearson_corr))
    with open("msm/q_vs_psi1_corr_TS.dat", "w") as fout:
        fout.write(str(pearson_corr_TS))
    with open("msm/ti_separation.dat", "w") as fout:
        fout.write(str(separation))
    with open("msm/spectral_gap_index.dat", "w") as fout:
        fout.write(str(max_sep))

