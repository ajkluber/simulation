import os
import glob
import numpy as np

#import simulation.calc.observables as observables

if __name__ == "__main__":
    with open("Qtanh_0_05_profile/T_used.dat", "r") as fin:
        T_used = float(fin.read())

    trajfiles = glob.glob("T_{:.2f}_1/traj.xtc".format(T_used)) + \
            glob.glob("T_{:.2f}_2/traj.xtc".format(T_used)) + \
            glob.glob("T_{:.2f}_3/traj.xtc".format(T_used))
    #trajfiles = [ x.rstrip("\n") for x in open("ticatrajs","r").readlines() ]
    tempdirs = [ x.split("/")[0] for x in trajfiles ]

#    dir = os.path.dirname(trajfiles[0])
#    pairs = np.loadtxt("%s/native_contacts.ndx" % dir, skiprows=1, dtype=int) - 1
#    n_native_pairs = pairs.shape[0]
#    r0 = np.loadtxt("%s/pairwise_params" % dir, usecols=(4,))[1:2*n_native_pairs:2]
#    r0_cont = r0 + 0.1
#    widths = 0.05
#    top = "%s/Native.pdb" % dir

    if all([ (os.path.exists(x + "/Rg.dat") & os.path.exists(x + "/Qtanh_0_05.dat")) \
                for x in tempdirs ]):
        qtanh = [ np.loadtxt(x + "/Qtanh_0_05.dat") for x in tempdirs ]
        rg = [ np.loadtxt(x + "/Rg.dat") for x in tempdirs ]
    elif all([ (os.path.exists(x + "/Rg.npy") & os.path.exists(x + "/Qtanh_0_05.npy")) \
                for x in tempdirs ]):
        qtanh = [ np.load(x + "/Qtanh_0_05.npy") for x in tempdirs ]
        rg = [ np.load(x + "/Rg.npy") for x in tempdirs ]
    else:
        raise IOError("Rg.dat, Rg.npy do not exist!")
        #qtanhsum_obs = observables.TanhContactSum(top, pairs, r0_cont, widths)
        #qtanh = observables.calculate_observable(trajfiles, qtanhsum_obs, saveas="Qtanh_0_05.dat")

    q = np.concatenate(qtanh)
    Rg = np.concatenate(rg)
    n, bin_edges = np.histogram(q, bins=40)
    mid_bin = 0.5*(bin_edges[1:] + bin_edges[:-1])

    Rg_bin_avg = np.zeros(len(bin_edges) - 1,float)
    dRg2_bin_avg = np.zeros(len(bin_edges) - 1,float)
    for i in range(len(bin_edges) - 1):
        frames_in_this_bin = ((q > bin_edges[i]) & (q <= bin_edges[i+1]))
        if any(frames_in_this_bin):
            Rg_bin_avg[i] = np.mean(Rg[frames_in_this_bin])
            dRg2_bin_avg[i] = np.mean((Rg[frames_in_this_bin] - Rg_bin_avg[i])**2)

    if not os.path.exists("Rg_vs_Qtanh_0_05"):
        os.mkdir("Rg_vs_Qtanh_0_05")
    os.chdir("Rg_vs_Qtanh_0_05")
    np.save("mid_bin.npy",mid_bin)
    np.save("Rg_vs_bin.npy",Rg_bin_avg)
    os.chdir("..")

    if not os.path.exists("dRg2_vs_Qtanh_0_05"):
        os.mkdir("dRg2_vs_Qtanh_0_05")
    os.chdir("dRg2_vs_Qtanh_0_05")
    np.save("mid_bin.npy",mid_bin)
    np.save("dRg2_vs_bin.npy",dRg2_bin_avg)
    os.chdir("..")
