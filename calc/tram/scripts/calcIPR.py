import numpy as np
import matplotlib.pyplot as plt

import simulation.calc.tram.util as util

if __name__ == "__main__":
    # the inverse participation ratio gives the effective number of states. It
    # can be calculated as the average Boltzmann weight,
    # IPR = \sum_i p_i p_i = <p_i>

    # IPR ranges from 1 to n_clusters. When IPR is equal to n_clusters then
    # each cluster is equally populated and we say the system is delocalized.
    # When IPR is equal to 1 the system is localized in one state. 

    coordfile = "Qtanh_0_05.npy"
    coordname = coordfile.split(".")[0]

    dirs, dtrajs, lagtimes, models = util.load_markov_state_models()

    if not os.path.exists("inverse_participation_ratio"):
        os.mkdir("inverse_participation_ratio")
    os.chdir("inverse_participation_ratio")

    model_msm = models[7]   # Lagtime of 200
    clust_idxs = np.arange(model_msm.nstates)
    pi = model_msm.stationary_distribution
    Neff = 1/np.dot(pi,pi)

    obs_trajs = [ np.load(x + "/" + coordfile) for x in dirs ]

    clust_obs = util.sort_observable_into_clusters(clust_idxs, obs_trajs, dtrajs)

    qavg_clust = np.array([ np.mean(x) for x in clust_obs ])
    
    # calculate inverse participation ratio for each stratum of Q.
    
    nbins = 10
    P_q, bins = np.histogram(qavg_clust, bins=nbins, density=True)
    P_q *= (bins[1] - bins[0])
    q_mid_bin = 0.5*(bins[1:] + bins[:-1])

    Neff_vs_Q = np.nan*np.zeros(nbins)
    for i in range(nbins - 1):
        in_bin = ((qavg_clust > bins[i]) & (qavg_clust <= bins[i + 1]))
        if np.any(in_bin):
            pi_q = pi[in_bin]/P_q[i]
            Neff_vs_Q[i] = 1./np.sum(pi_q**2)

    plt.figure()
    plt.plot(q_mid_bin, Neff_vs_Q)
    plt.xlabel("$Q$")
    plt.ylabel("$N_{eff}(Q)$ states")
    plt.title("$N_{eff} = {}$ ($N_{tot} = 1000$)".format(Neff))
    #plt.savefig("")

    #with open("inv_participation", "w") as fout:
    #    fout.write()

    # Save:
    # - overall inverse participation ratio
    # - IPR vs Q 

    os.chdir("..")
