import os
import numpy as np 

import simulation.calc.tram.util as util

def get_number_of_dominant_timescales(lagtime, lag_idx):
    if os.path.exists("msm/timescales.npy"):
        imp_ti = np.load("msm/timescales.npy")[lag_idx]
    else:
        # Load MSM's that have already been calculated.
        dirs, dtrajs, lagtimes, models = util.load_markov_state_models()

        # lagtime of 200
        tau = lagtimes[lag_idx]
        model_msm = models[lag_idx] 
        imp_ti = tau*model_msm.timescales()[:10]

    # How do we determine the number of slow timescales?
    # Determine where timescales end up getting ''bunched''? Spectral gap?

    phys_imp_ti = imp_ti[imp_ti >= 1.05*lagtime]

    if len(phys_imp_ti) < 3:
        n_dominant = len(phys_imp_ti)
    else:
        # ratio of timescale with next timescale.
        ratio = phys_imp_ti[:-1]/phys_imp_ti[1:]
        n_dominant = np.argmax(ratio) + 1

    with open("msm/n_dominant_timescales.dat", "w") as fout:
        fout.write(str(n_dominant))

    return n_dominant

def get_number_of_meaningful_timescales(lagtime, lag_idx):
    if os.path.exists("msm/timescales.npy"):
        imp_ti = np.load("msm/timescales.npy")[lag_idx]
    else:
        # Load MSM's that have already been calculated.
        dirs, dtrajs, lagtimes, models = util.load_markov_state_models()

        # lagtime of 200
        tau = lagtimes[lag_idx]
        model_msm = models[lag_idx] 
        imp_ti = tau*model_msm.timescales()[:10]

    n_physical = np.sum(imp_ti >= 1.05*lagtime)

    with open("msm/n_physical_timescales.dat", "w") as fout:
        fout.write(str(n_physical))

    return n_physical

if __name__ == "__main__":
    pass

    # Determine the number of clusters by the number of slow timescales.
    #n_pcca = np.sum(model_msm.timescales() >= 1) + 1

    #with open("msm/n_metastable") as fout:
    #    fout.write("{}".format(n_pcca))

