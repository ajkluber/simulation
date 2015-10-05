import os
import argparse
import numpy as np

import simulation.calc.util as util

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

    parser.add_argument('--n_bins',
            type=int,
            default=40,
            help='Contact functional form.')

    parser.add_argument('--topology',
            type=str,
            default="Native.pdb",
            help='Contact functional form. Opt.')

    parser.add_argument('--tanh_scale',
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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    trajsfile = args.trajs
    function = args.function 
    coordfile = args.coordfile
    coordname = coordfile.split(".")[0]
    n_bins = args.n_bins
    chunksize = args.chunksize
    topology = args.topology
    periodic = args.periodic
 
    # Data source
    trajfiles = [ "%s" % x.rstrip("\n") for x in open(trajsfile,"r").readlines() ]
    dir = os.path.dirname(trajfiles[0])

    if args.saveas is None:
        save_coord_as = {"step":"Q.dat","tanh":"Qtanh.dat","w_tanh":"Qtanh_w.dat"}[function]
    else:
        save_coord_as = args.saveas

    # Parameterize contact-based reaction coordinate
    contact_params = util.get_contact_params(dir,args)
    coord_sources = [  "%s/%s" % (os.path.dirname(trajfiles[i]),coordfile) for i in range(len(trajfiles)) ]

    if not all([ os.path.exists(x) for x in coord_sources ]):
        # Parameterize contact-based reaction coordinate
        contact_function = util.get_sum_contact_function(pairs,function,contact_params,periodic=periodic)

        # Calculate contact function over directories
        contacts = util.calc_coordinate_multiple_trajs(trajfiles,contact_function,topology,chunksize,save_coord_as=args.coordfile,collect=True)
    else:
        # Load precalculated coordinate
        contacts = [ np.loadtxt(coord_sources[i]) for i in range(len(coord_sources)) ]

    # Parameterize pairwise contact function
    pairwise_contact_function = util.get_pair_contact_function(pairs,function,contact_params,periodic=periodic)

    # Calculate pairwise contacts over directories 
    bin_edges,avgQi_by_bin = util.bin_multiple_coordinates_for_multiple_trajs(trajfiles,
            contacts,pairwise_contact_function,pairs.shape[0],n_bins,topology,chunksize)

    # Save  
    #np.savetxt("",avgQi_by_bin)
    #np.savetxt("bin_edges.dat",bin_edges)
    #np.savetxt("mid_bin.dat",mid_bin)

