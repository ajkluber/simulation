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

    parser.add_argument('--function',
            type=str,
            required=True,
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

    parser.add_argument('--saveas',
            type=str,
            help='File to save in directory.')

    parser.add_argument('--savepath',
            type=str,
            help='Directory to save in.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    trajsfile = args.trajs
    function = args.function 
    topology = args.topology
    chunksize = args.chunksize
    periodic = args.periodic

    # Data source
    trajfiles = [ "%s" % x.rstrip("\n") for x in open(trajsfile,"r").readlines() ]
    dir = os.path.dirname(trajfiles[0])

    if args.saveas is None:
        save_coord_as = {"step":"Q.dat","tanh":"Qtanh.dat","w_tanh":"Qtanh_w.dat"}[function]
    else:
        save_coord_as = args.saveas

    # Parameterize contact-based reaction coordinate
    pairs, contact_params = util.get_contact_params(dir,args)
    contact_function = util.get_sum_contact_function(pairs,function,contact_params,periodic=periodic)

    # Calculate contact function over directories
    util.calc_coordinate_multiple_trajs(trajfiles,contact_function,topology,chunksize,save_coord_as=save_coord_as,savepath=args.savepath)

