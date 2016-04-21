import os
import argparse
import numpy as np

import simulation.calc.binned_contacts
import simulation.calc.util as util
import simulation.calc.pmfutil as pmfutil

def TS_probabilities(Tdirs,coordfile,contact_args):
    """ Calculate the TS contact probabilities along some reaction coordinate

    Parameters
    ----------

    Tdirs : list
        List of directory names to collect reaction coordinate file from

    coordfile : str
        Name of file containing reaction coordinate.

    contact_args : object
        Object containing the 

    """

    T = Tdirs[0].split("_")[0]
    coordname = coordfile.split(".")[0]
    if not os.path.exists("%s_profile/state_bounds.txt" % coordname):
        # Determine state bounds from 1D profile
        print "calculating state bounds"
        coordvst = np.concatenate([np.loadtxt("%s/%s" % (x,coordfile)) for x in Tdirs ])
        if not os.path.exists("%s_profile" % coordname):
            os.mkdir("%s_profile" % coordname)
        os.chdir("%s_profile" % coordname)
        mid_bin, Fdata = pmfutil.pmf1D(coordvst,bins=40)
        xinterp, F = pmfutil.interpolate_profile(mid_bin,Fdata)
        minidx, maxidx = pmfutil.extrema_from_profile(xinterp,F)
        min_bounds, max_bounds = pmfutil.state_bounds_from_profile(xinterp,F)
        min_labels, max_labels = pmfutil.assign_state_labels(min_bounds,max_bounds)
        pmfutil.save_state_bounds(T,min_bounds,max_bounds,min_labels,max_labels)
        os.chdir("..")
        with open("%s_profile/%s_state_bounds.txt" % (coordname,T),"r") as fin:
            state_bins = np.array([ [float(x.split()[1]),float(x.split()[2])] 
                for x in fin.readlines() if x.split()[0].startswith("TS")])
    else:
        # Load state bounds
        with open("%s_profile/state_bounds.txt" % coordname,"r") as fin:
            state_bins = np.array([ [float(x.split()[1]),float(x.split()[2])] 
                for x in fin.readlines() if x.split()[0].startswith("TS")])

    # Calculate the contact probability in TS
    print "calculating TS"
    contact_args.bins = state_bins
    bin_edges, qi_vs_Q = simulation.calc.binned_contacts.calculate_binned_contacts_vs_q(contact_args)
    TS = qi_vs_Q[0,:]
    return TS

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

