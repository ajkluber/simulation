import os
import numpy as np

import simulation.calc.tram.util as util

if __name__ == "__main__":
    # the only physically meanginful timescales are longer than the lagtime
    tau = 200
    lag_idx = 7

    dirs, dtrajs, lagtimes, models = util.load_markov_state_models()

    M = models[7]   # Lagtime of 200
    ti = tau*M.timescales()
    #ti_phys = ti
    ti_phys = ti[ti >= tau]

    if len(ti_phys) == 1:
        KC = np.ones(ti_phys.shape[0], float)
    else:
        KC = np.array([ np.sum(ti_phys[:i])/np.sum(ti_phys) for i in range(1,len(ti_phys) + 1) ])

    np.save("msm/KC.npy", KC)
    #plt.plot(np.arange(1, len(KC) + 1), KC, 'o')
    #plt.plot(np.arange(1, len(KC) + 1), KC)
    #plt.ylim(0, 1)
    #plt.show()

