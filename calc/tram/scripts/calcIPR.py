import os
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

    M = models[7]   # Lagtime of 200
    clust_idxs = np.arange(M.nstates)
    pi = M.stationary_distribution
    Neff = 1/np.dot(pi,pi)

    np.save("msm/Neff.npy", np.array([Neff]))

    # Get q on clusters
    obs_trajs = [ np.load(x + "/" + coordfile) for x in dirs ]
    clust_obs = util.sort_observable_into_clusters(clust_idxs, obs_trajs, dtrajs)
    qavg_clust = np.array([ np.mean(x) for x in clust_obs ])


    psi1 = M.eigenvectors_right[:,1] 
    psi2 = M.eigenvectors_right[:,2] 


    corr_q_psi1 = np.dot(psi1, qavg_clust)/(np.linalg.norm(psi1)*np.linalg.norm(qavg_clust))
    corr_q_psi2 = np.dot(psi2, qavg_clust)/(np.linalg.norm(psi2)*np.linalg.norm(qavg_clust))

    # correlation coefficient of q with psi1
    plt.figure()
    plt.plot(qavg_clust, psi1, '.')
    


    
    
    raise SystemExit

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

    # Effective number of states per layer.
    Neff_vs_Q2 = np.nan*np.zeros(nbins)
    for i in range(nbins - 1):
        in_bin = ((qavg_clust > bins[i]) & (qavg_clust <= bins[i + 1]))
        if np.any(in_bin):
            pi_tilde = pi[in_bin]/np.sum(pi[in_bin])
            Neff_vs_Q2[i] = 1./np.dot(pi_tilde, pi_tilde)

    # Number of clusters per layer
    N_vs_Q = np.nan*np.zeros(nbins)
    for i in range(nbins - 1):
        N_vs_Q[i] = np.sum(((qavg_clust > bins[i]) & (qavg_clust <= bins[i + 1])))

    if not os.path.exists("inverse_participation_ratio"):
        os.mkdir("inverse_participation_ratio")
    os.chdir("inverse_participation_ratio")

    plt.figure()
    plt.plot(q_mid_bin, Neff_vs_Q2)
    plt.xlabel("$Q$")
    plt.ylabel("$N_{eff}(Q)$ states")
    plt.title("$N_{{eff}} = {}$ ($N_{{tot}} = 1000$)".format(Neff))

    plt.figure()
    plt.plot(q_mid_bin, Neff_vs_Q)
    plt.xlabel("$Q$")
    plt.ylabel("$N_{eff}(Q)$ states")
    plt.title("$N_{{eff}} = {}$ ($N_{{tot}} = 1000$)".format(Neff))
    #plt.savefig("Neff_vs_Q.pdf")
    #plt.savefig("Neff_vs_Q.png")
    plt.show()

    #with open("inv_participation", "w") as fout:
    #    fout.write()

    # Save:
    # - overall inverse participation ratio
    # - IPR vs Q 

    os.chdir("..")
