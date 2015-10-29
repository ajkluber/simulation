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

    parser.add_argument('--contacts',
            type=str,
            required=True,
            help='Compute energy for native or non-native contacts?')

    parser.add_argument('--topology',
            type=str,
            default="Native.pdb",
            help='Contact functional form. Opt.')

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
            help='Filename to save coordinate as.')

    parser.add_argument('--savepath',
            type=str,
            help='Directory to save in.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    trajsfile = args.trajs
    contacts = args.contacts
    topology = args.topology
    chunksize = args.chunksize
    periodic = args.periodic
    
    if contacts not in ["native","nonnative"]:
        raise IOError("--contacts must be: native OR nonnative")

    if args.saveas is None:
        save_coord_as = {"native":"Enative.dat","nonnative":"Enonnative.dat"}[contacts]
    else:
        save_coord_as = args.saveas

    # Data source
    trajfiles = [ "%s" % (x.rstrip("\n")) for x in open(trajsfile,"r").readlines() ]
    dir = os.path.dirname(trajfiles[0])

    
    util.get_pair_energy_params(dir)


    # Parameterize contact energy function.
    energy_function = util.get_contact_energy_function(pairs,pair_type,eps,contact_params,periodic=periodic)

    # Calculate contact function over directories
    util.calc_coordinate_multiple_trajs(trajfiles,energy_function,topology,chunksize,save_coord_as=save_coord_as,savepath=args.savepath)
