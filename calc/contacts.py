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

    util.check_if_supported(function)

    if args.saveas is None:
        save_coord_as = {"step":"Q.dat","tanh":"Qtanh.dat","w_tanh":"Qtanh_w.dat"}[function]
    else:
        save_coord_as = args.saveas

    # Data source
    trajfiles = [ "%s" % x.rstrip("\n") for x in open(trajsfile,"r").readlines() ]
    dir = os.path.dirname(trajfiles[0])
    n_native_pairs = len(open("%s/native_contacts.ndx" % dir).readlines()) - 1
    if os.path.exists("%s/pairwise_params" % dir):
        r0 = np.loadtxt("%s/pairwise_params" % dir,usecols=(4,),skiprows=1)[1:2*n_native_pairs:2] + 0.1
    elif os.path.exists("%s/native_contact_distances.dat" % dir):
        r0 = np.loadtxt("%s/native_contact_distances.dat" % dir) + 0.1
    else:
        raise IOError("Need source for native contact distances!")
    assert r0.shape[0] == n_native_pairs

    # Get contact function parameters
    if function == "w_tanh":
        if (not os.path.exists(args.tanh_weights)) or (args.tanh_weights is None):
            raise IOError("Weights file doesn't exist: %s" % args.tanh_weights)
        else:
            pairs = np.loadtxt(args.tanh_weights,usecols=(0,1),dtype=int)
            widths = args.tanh_scale*np.ones(pairs.shape[0],float)
            weights = np.loadtxt(args.tanh_weights,usecols=(2,),dtype=float)
            contact_params = (r0,widths,weights)
    elif function == "tanh":
        pairs = np.loadtxt("%s/native_contacts.ndx" % dir,skiprows=1,dtype=int) - 1
        widths = args.tanh_scale*np.ones(r0.shape[0],float)
        contact_params = (r0,widths)
    elif function == "step":
        pairs = np.loadtxt("%s/native_contacts.ndx" % dir,skiprows=1,dtype=int) - 1
        contact_params = (r0)
    else:
        raise IOError("--function must be in: %s" % util.supported_functions.keys().__str__())

    # Parameterize contact-based reaction coordinate
    contact_function = util.get_sum_contact_function(pairs,function,contact_params,periodic=periodic)

    # Calculate contact function over directories
    util.calc_coordinate_multiple_trajs(trajfiles,contact_function,topology,chunksize,save_coord_as=save_coord_as,savepath=args.savepath)

