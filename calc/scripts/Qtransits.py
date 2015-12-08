import numpy as np
import os

import simulation.calc.transits as transits

if __name__ == "__main__":
    trajfiles = [ x.rstrip("\n") for x in open("ticatrajs","r").readlines() ]
    dirs = [ os.path.dirname(x) for x in trajfiles ]

    coordfile = "Qtanh_0_05.dat"
    coordname = coordfile.split(".dat")[0]

    # get location of minima according to free energy profile.
    if os.path.exists("%s_profile/minima.dat" % coordname):
        minima = np.loadtxt("%s_profile/minima.dat" % coordname)
        U = minima.min()
        N = minima.max()
    else:
        raise IOError("%s_profile/minima.dat does not exist!" % coordname)

    # calculate transit time between minima for each trajectory.
    all_dwellsU = []
    all_dwellsN = []
    all_transitsUN = []
    all_transitsNU = []
    for i in range(len(dirs)):
        os.chdir(dirs[i])
        x = np.loadtxt(coordfile)
        dtraj = np.zeros(x.shape[0], int)
        dtraj[x <= U] = 0
        dtraj[x >= N] = 2
        dtraj[(x > U) & (x < N)] = 1

        # calculate transits
        dwellsU, dwellsN, transitsUN, transitsNU = transits.partition_dtraj(dtraj, 0, 2)

        # save transitition path details.
        #if not os.path.exists("%s_transits" % coordname):
        #    os.mkdir("%s_transits" % coordname)
        #os.chdir("%s_transits" % coordname)
        #np.savetxt("dwellU.dat", dwellsU, fmt="%5d")
        #np.savetxt("dwellN.dat", dwellsN, fmt="%5d")
        #np.savetxt("transitsUN.dat", transitsUN, fmt="%5d")
        #np.savetxt("transitsNU.dat", transitsNU, fmt="%5d")
        #os.chdir("..")
        os.chdir("..")

        all_dwellsU.append(dwellsU)
        all_dwellsN.append(dwellsN)
        all_transitsUN.append(transitsUN)
        all_transitsNU.append(transitsNU)
        

    folding_time = np.concatenate([ all_dwellsU[i][:,1] + all_transitsUN[i][:,1] for i in range(len(all_dwellsU)) ])
    unfolding_time = np.concatenate([ all_dwellsN[i][:,1] + all_transitsNU[i][:,1] for i in range(len(all_dwellsN)) ])

    forward_transit_time = np.concatenate([ all_transitsUN[i][:,1] for i in range(len(all_transitsUN)) ])
    backward_transit_time = np.concatenate([ all_transitsNU[i][:,1] for i in range(len(all_transitsNU)) ])
    
    # Save transit times
    if not os.path.exists("%s_transit_time" % coordname): 
        os.mkdir("%s_transit_time" % coordname)
    os.chdir("%s_transit_time" % coordname)
    with open("forward_mean","w") as fout:
        fout.write("%.2f" % np.mean(forward_transit_time))
    with open("backward_mean","w") as fout:
        fout.write("%.2f" % np.mean(backward_transit_time))
    np.savetxt("forward_transit_times.dat", forward_transit_time, fmt="%5d")
    np.savetxt("backward_transit_times.dat", backward_transit_time, fmt="%5d")
    os.chdir("..")

    # Save folding times
    if not os.path.exists("%s_folding_time" % coordname): 
        os.mkdir("%s_folding_time" % coordname)
    os.chdir("%s_folding_time" % coordname)
    with open("folding_mean","w") as fout:
        fout.write("%.2f" % np.mean(folding_time))
    with open("unfolding_mean","w") as fout:
        fout.write("%.2f" % np.mean(unfolding_time))
    np.savetxt("folding_times.dat", folding_time, fmt="%5d")
    np.savetxt("unfolding_times.dat", unfolding_time, fmt="%5d")
    os.chdir("..")
