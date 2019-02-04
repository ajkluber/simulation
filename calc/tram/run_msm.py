import os
import pickle
import argparse
import numpy as np



#def test_custom_feature():
#    """Simple test for the custom feature"""
#    from pyemma.coordinates.data.featurization.misc import CustomFeature
#    def tanh_contact(traj, pairs, r0, widths):
#        r = md.compute_distances(traj, pairs)
#        return 0.5*(np.tanh((r0 - r)/widths) + 1)
#    pairs = np.array([[1,10], [1,20], [5, 40], [12, 15], [12, 40]]) - 1
#    r0 = np.ones(len(pairs), np.float32)*0.5
#    widths = np.ones(len(pairs), np.float32)*0.1
#    feat.add_custom_feature(CustomFeature(tanh_contact, pairs, r0, widths, dim=len(pairs)))

def run_msm_for_directory(topfile, trajfiles, savedir, T, cwd):
    """ """

    # add features
    feat = coor.featurizer(topfile)
    feat, feature_info = util.sbm_contact_features(feat, pairwise_file, n_native_pairs)

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    if (not os.path.exists("{}/dtrajs.pkl".format(savedir))) or recluster:
        # cluster if necessary
        reader = coor.source(trajfiles, feat)
        transform = coor.tica(dim=tica_dims, lag=tica_lag, stride=stride)
        cluster = coor.cluster_kmeans(k=n_clusters)
        pipeline = coor.pipeline([reader, transform, cluster])
        dtrajs = cluster.dtrajs

        os.chdir(savedir)
        dirs = [ os.path.basename(os.path.dirname(x)) for x in trajfiles ]

        if not dontsavemsm:
            dtraj_info = { dirs[x]:dtrajs[x] for x in range(len(dirs)) }
            dtraj_info["dirs"] = dirs
            with open("dtrajs.pkl", 'wb') as fhandle:
                pickle.dump(dtraj_info, fhandle)

    else:
        os.chdir(savedir)
        with open("dtrajs.pkl", 'rb') as fhandle:
            dtraj_pkl = pickle.load(fhandle)
            dirs = dtraj_pkl["dirs"]
            dtrajs = [ dtraj_pkl[x] for x in dirs ]

    # estimate MSM's at different lagtimes
    #lags = [1,2,5,10,20,50,100,200,300,400,500,600,700,800,900,1000]
    lags = [1,2,5,10,20,50,100,500,1000,2000,5000,10000,20000]
    its = msm.its(dtrajs, lags=lags)

    if not dontsavemsm:
        util.save_markov_state_models(T, its.models)

    mplt.plot_implied_timescales(its, ylog=False)
    plt.title("T = " + str(T))
    plt.savefig("its_tanh_cont_features.pdf")
    plt.savefig("its_tanh_cont_features.png")

    os.chdir(cwd)
    
    if display:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pairwise_file")
    parser.add_argument("n_native_pairs", type=int)
    parser.add_argument("tica_lag", type=int)
    parser.add_argument("--trajname", default="traj.xtc", type=str)
    parser.add_argument("--topname", default="ref.pdb", type=str)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--n_clusters", default=400, type=int)
    parser.add_argument("--tica_dims", default=10, type=int)
    parser.add_argument("--cold_temps", action="store_true")
    parser.add_argument("--recluster", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--dontsavemsm", action="store_true")
    args = parser.parse_args()

    pairwise_file = args.pairwise_file
    n_native_pairs = args.n_native_pairs
    tica_lag = args.tica_lag
    trajname = args.trajname
    topname = args.topname
    stride = args.stride
    n_clusters = args.n_clusters
    tica_dims = args.tica_dims
    cold_temps = args.cold_temps
    recluster = args.recluster
    display = args.display
    dontsavemsm = args.dontsavemsm

    # for plotting on compute node
    if display:
        import pyemma.plots as mplt
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pyemma.plots as mplt

    import pyemma.coordinates as coor
    import pyemma.msm as msm

    import util
    import simulation.calc.util

    cwd = os.getcwd()

    if cold_temps:
        organized_temps = simulation.calc.util.get_organized_temps("cold_temps")
        T = organized_temps.keys()
        T.sort()

        for i in range(len(T)):
            tempdirs = organized_temps[T[i]] 
            topfile = tempdirs[0] + "/" + topname
            trajfiles = [ x + "/" + trajname for x in tempdirs ]
            savedir = "{}/msm".format(tempdirs[0])

            run_msm_for_directory(topfile, trajfiles, savedir, T[i], cwd)
    else:
        with open("Qtanh_0_05_profile/T_used.dat","r") as fin: 
            T = float(fin.read())
        tempdirs = [ "T_{:.2f}_{}".format(T, x) for x in [1,2,3] ]
        topfile = tempdirs[0] + "/" + topname
        trajfiles = [ x + "/" + trajname for x in tempdirs ]
        savedir = "msm"

        run_msm_for_directory(topfile, trajfiles, savedir, T, savedir)


