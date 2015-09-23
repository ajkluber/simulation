import os
import argparse
import numpy as np

import simulation.calc.util as util

def get_args():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--dirs',
            type=str,
            required=True,
            help='File holding directory names.')

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
            help='File to save in directory.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    dirsfile = args.dirs
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
    cwd = os.getcwd()
    trajfiles = [ "%s/%s/traj.xtc" % (cwd,x.rstrip("\n")) for x in open(dirsfile,"r").readlines() ]
    dir = os.path.dirname(trajfiles[0])
    n_native_pairs = len(open("%s/native_contacts.ndx" % dir).readlines()) - 1
    if (not os.path.exists("%s/pairwise_params" % dir)) or (not os.path.exists("%s/model_params" % dir)):
        raise IOError("%s/pairwise_params or %s/model_params does not exist!" % (dir,dir))
    else:
        # Get potential parameters. 
        # Assumes gaussian contacts.
        # Doesn't include exluded volume terms.
        if contacts == "native":
            pairs = np.loadtxt("%s/pairwise_params" % dir,usecols=(0,1),skiprows=1,dtype=int)[1:2*n_native_pairs + 1:2] - 1
            param_idx = np.loadtxt("%s/pairwise_params" % dir,usecols=(2,),skiprows=1,dtype=int)[1:2*n_native_pairs + 1:2]
            pair_type = np.loadtxt("%s/pairwise_params" % dir,usecols=(3,),skiprows=1,dtype=int)[1:2*n_native_pairs + 1:2]
            eps = np.loadtxt("%s/model_params" % dir,skiprows=1)[1:2*n_native_pairs + 1:2]

            contact_params = [] 
            with open("%s/pairwise_params" % dir,"r") as fin:
                all_lines = fin.readlines()[2:2*n_native_pairs + 2:2]
                for i in range(pairs.shape[0]):
                    temp_params = all_lines[i].rstrip("\n").split()[4:]
                    contact_params.append(tuple([ float(x) for x in temp_params]))
        else:
            pairs = np.loadtxt("%s/pairwise_params" % dir,usecols=(0,1),skiprows=1,dtype=int)[2*n_native_pairs + 1::2] - 1
            param_idx = np.loadtxt("%s/pairwise_params" % dir,usecols=(2,),skiprows=1,dtype=int)[2*n_native_pairs + 1::2]
            pair_type = np.loadtxt("%s/pairwise_params" % dir,usecols=(3,),skiprows=1,dtype=int)[2*n_native_pairs + 1::2]
            eps = np.loadtxt("%s/model_params" % dir,skiprows=1)[2*n_native_pairs + 1::2]

            contact_params = [] 
            with open("%s/pairwise_params" % dir,"r") as fin:
                all_lines = fin.readlines()[2*n_native_pairs + 2::2]
                for i in range(pairs.shape[0]):
                    temp_params = all_lines[i].rstrip("\n").split()[4:]
                    contact_params.append(tuple([ float(x) for x in temp_params]))

    # Parameterize contact energy function.
    energy_function = util.get_contact_energy_function(pairs,pair_type,eps,contact_params,periodic=periodic)

    # Calculate contact function over directories
    util.calc_coordinate_multiple_trajs(trajfiles,energy_function,topology,chunksize,save_coord_as=save_coord_as)
