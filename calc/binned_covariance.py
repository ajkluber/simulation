import os
import argparse
import numpy as np

import simulation.calc.util as util
import simulation.calc.binned_contacts as binned_contacts

def calculate_binned_contacts_vs_q(args):
    trajsfile = args.trajs
    function = args.function 
    coordfile = args.coordfile
    coordname = coordfile.split(".")[0]
    bins = args.bins
    chunksize = args.chunksize
    topology = args.topology
    periodic = args.periodic
 
    # Data source
    trajfiles = [ "%s" % x.rstrip("\n") for x in open(trajsfile,"r").readlines() ]
    dir = os.path.dirname(trajfiles[0])

    # Parameterize contact-based reaction coordinate
    pairs, contact_params = util.get_contact_params(dir,args)
    coord_sources = [  "%s/%s" % (os.path.dirname(trajfiles[i]),coordfile) for i in range(len(trajfiles)) ]

    if not all([ os.path.exists(x) for x in coord_sources ]):
        # Parameterize contact-based reaction coordinate
        contact_function = util.get_sum_contact_function(pairs,function,contact_params,periodic=periodic)

        # Calculate contact function over directories
        contacts = util.calc_coordinate_multiple_trajs(trajfiles,contact_function,topology,chunksize,save_coord_as=args.coordfile,savepath=args.savepath,collect=True)
    else:
        # Load precalculated coordinate
        contacts = [ np.loadtxt(coord_sources[i]) for i in range(len(coord_sources)) ]

    # Parameterize pairwise contact function
    pairwise_contact_function = util.get_pair_contact_function(pairs,function,contact_params,periodic=periodic)

    # Calculate pairwise contacts over directories 
    bin_edges,avgqi_by_bin = util.bin_multiple_coordinates_for_multiple_trajs(trajfiles,
            contacts,pairwise_contact_function,pairs.shape[0],bins,topology,chunksize)

    return bin_edges, avgqi_by_bin

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

    parser.add_argument('--function',
            type=str,
            required=True,
            help='Contact functional form.')

    parser.add_argument('--bins',
            type=int,
            default=40,
            help='Contact functional form.')

    parser.add_argument('--topology',
            type=str,
            default="Native.pdb",
            help='Contact functional form. Opt.')

    parser.add_argument('--tanh_scale',
            type=float,
            default=0.3,
            help='Tanh contact switching scale. Opt.')

    parser.add_argument('--tanh_weights',
            type=float,
            help='Tanh contact weights. Opt.')

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
    coordname = coordfile.split(".")[0]
 
    bin_edges, avgqi_by_bin = binned_contacts.calculate_binned_contacts_vs_q(args)


    # parameterize

    #bin_edges, avgqi_by_bin = calculate_binned_energy_vs_q(args)

    cwd = os.getcwd()
    if args.savepath is not None:
        os.chdir(args.savepath)

    # Save  
    if not os.path.exists("binned_contacts_vs_%s" % coordname):
        os.mkdir("binned_contacts_vs_%s" % coordname)
    os.chdir("binned_contacts_vs_%s" % coordname)
    np.savetxt("contacts_vs_bin.dat",avgqi_by_bin)
    np.savetxt("bin_edges.dat",bin_edges)
    os.chdir(cwd)
    print "Took: %.2f" % ((time.time() - starttime)/60.)
