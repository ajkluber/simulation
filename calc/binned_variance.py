import os
import argparse
import logging
import numpy as np

import simulation.calc.util as util

def setup_logger(args):
    """Log starting information"""
    cwd = os.getcwd()
    if args.savepath is not None:
        os.chdir(args.savepath)
    if not os.path.exists("binned_d%s2_vs_%s" % (args.varcoordname,args.bincoordname)):
        os.mkdir("binned_d%s2_vs_%s" % (args.varcoordname,args.bincoordname))

    os.chdir("binned_d%s2_vs_%s" % (args.varcoordname,args.bincoordname))
    logging.basicConfig(filename="d%s2_bin_avg.log" % args.varcoordname,
                        filemode="w",
                        format="%(levelname)s:%(name)s:%(asctime)s: %(message)s",
                        datefmt="%H:%M:%S",
                        level=logging.DEBUG)
    logger = logging.getLogger('binned_variance')
    logger.info("data source          = %s" % args.dir)
    logger.info("input parameters:")
    logger.info("  trajsfile          = %s" % args.trajs)
    logger.info("  trajfiles          = %s" % args.trajfiles.__str__())
    logger.info("  binning_coordinate = %s" % args.coordfile)
    logger.info("  bin_coord_name     = %s" % args.bincoordname)
    logger.info("  var_coord_name     = %s" % args.varcoordname)
    logger.info("  bins               = %s" % args.bins.__str__())
    logger.info("  topology           = %s" % args.topology)
    logger.info("  chunksize          = %s" % args.chunksize)
    logger.info("  periodic           = %s" % args.periodic)
    os.chdir(cwd)
    return logger

def get_args():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--trajs',
            type=str,
            required=True,
            help='File holding trajectory paths.')

    parser.add_argument('--coordfile',
            type=str,
            required=True,
            help='File of reaction coordinate.')

    parser.add_argument('--contacts',
            type=str,
            required=True,
            help='Compute energy for native or non-native contacts?')

    parser.add_argument('--bins',
            type=int,
            default=40,
            help='Number of bins along binning coordinate.')

    parser.add_argument('--topology',
            type=str,
            default="Native.pdb",
            help='Filename for MDTraj topology info. (pdb) Opt.')

    parser.add_argument('--chunksize',
            type=int,
            default=1000,
            help='Chunk size to parse traj.')

    parser.add_argument('--periodic',
            type=bool,
            default=False,
            help='Periodic.')

    parser.add_argument('--savepath',
            type=str,
            help='Directory to save in.')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    import time
    starttime = time.time()

    args = get_args()

    trajsfile = args.trajs
    coordfile = args.coordfile
    bins = args.bins
    topology = args.topology
    chunksize = args.chunksize
    periodic = args.periodic
    bincoordname = coordfile.split(".")[0]
    n_obs = 1
    n_obs1 = 1
    n_obs2 = 1

    varcoordname = {"native":"Enative","nonnative":"Enonnative"}[args.contacts]

    trajfiles = [ "%s" % (x.rstrip("\n")) for x in open(trajsfile,"r").readlines() ]
    dir = os.path.dirname(trajfiles[0])
    args.dir = dir
    args.trajfiles = trajfiles
    args.varcoordname = varcoordname
    args.bincoordname = bincoordname

    logger = setup_logger(args)

    # Parameterize contact energy function.
    logger.info("parameterize pair energy function")
    pairs,pair_type,eps,contact_params = util.get_pair_energy_params(dir,args)
    energy_function = util.get_contact_energy_function(pairs,pair_type,eps,contact_params,periodic=periodic)

    # TODO: compute binning coordinate on the fly if needed.
    logger.info("loading binning coordinate")
    coord_sources = [  "%s/%s" % (os.path.dirname(trajfiles[i]),coordfile) for i in range(len(trajfiles)) ]
    binning_coord = [ np.loadtxt(coord_sources[i]) for i in range(len(coord_sources)) ]

    if os.path.exists("binned_%s_vs_%s/%s_vs_bin.dat" % (varcoordname,bincoordname,varcoordname)):
        # Load binned energy
        logger.info("loading E_bin_avg")
        os.chdir("binned_%s_vs_%s" % (varcoordname,bincoordname))
        E_bin_avg = np.loadtxt("%s_vs_bin.dat" % varcoordname)
        bin_edges = np.loadtxt("bin_edges.dat",bin_edges)
        os.chdir("..")
    else:
        # Calculate binned energy
        logger.info("calculating E_bin_avg")
        bin_edges, E_bin_avg = util.bin_multiple_coordinates_for_multiple_trajs(trajfiles,binning_coord,energy_function,n_obs,bins,topology,chunksize)
        print "calculation took: %.2f" % ((time.time() - starttime)/60.)

        # Save  
        if args.savepath is not None:
            os.chdir(args.savepath)
        logger.info("saving E_bin_avg")
        if not os.path.exists("binned_%s_vs_%s" % (varcoordname,bincoordname)):
            os.mkdir("binned_%s_vs_%s" % (varcoordname,bincoordname))
        os.chdir("binned_%s_vs_%s" % (varcoordname,bincoordname))
        np.savetxt("%s_vs_bin.dat" % varcoordname,E_bin_avg)
        np.savetxt("bin_edges.dat",bin_edges)
        os.chdir(cwd)

    # Calculate binned variance
    logger.info("calculating dE2_bin_avg")
    dE2_bin_avg = util.bin_covariance_multiple_coordinates_for_multiple_trajs(trajfiles,binning_coord,
        energy_function,energy_function,E_bin_avg,E_bin_avg,
        n_obs1,n_obs2,bin_edges,topology,chunksize)

    # Save  
    if args.savepath is not None:
        os.chdir(args.savepath)
    if not os.path.exists("binned_d%s2_vs_%s" % (varcoordname,bincoordname)):
        os.mkdir("binned_d%s2_vs_%s" % (varcoordname,bincoordname))
    logger.info("saving dE2_bin_avg")
    os.chdir("binned_d%s2_vs_%s" % (varcoordname,bincoordname))
    for i in range(dE2_bin_avg.shape[0]):
        np.savetxt("d%s2_vs_bin_%d.dat" % (varcoordname,i),dE2_bin_avg[i])
    np.savetxt("bin_edges.dat",bin_edges)
    os.chdir(cwd)
    logger.info("finished")
