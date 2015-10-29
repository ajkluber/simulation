import os
import numpy as np

import mdtraj as md

class ContactArgs(object):
    """Class to parameterize contact function"""
    def __init__(self,trajsfile,bins=40,function="tanh",tanh_scale=0.05,
            coordfile="Qtanh_0_05.dat",topology="Native.pdb",chunksize=1000,periodic=False):
        self.trajs = trajsfile
        self.function = function
        self.coordfile = coordfile
        self.bins = bins
        self.tanh_scale = tanh_scale
        self.chunksize = chunksize
        self.topology = topology
        self.periodic = periodic

######################################################################
# Utility functions
######################################################################
def check_if_supported(function_type):
    if function_type not in supported_functions.keys():
        raise IOError("--function_type must be in " + supported_functions.__str__())

def get_sum_contact_function(pairs,function_type,contact_params,periodic=False):
    """Returns a function that takes a MDTraj Trajectory object"""
    contact_function = supported_functions[function_type]
    def obs_function(trajchunk):
        r = md.compute_distances(trajchunk,pairs,periodic=periodic)
        if type(contact_params) == tuple:
            return np.sum(contact_function(r,*contact_params),axis=1)
        else:
            return np.sum(contact_function(r,contact_params),axis=1)
    return obs_function

def get_pair_contact_function(pairs,function_type,contact_params,periodic=False):
    """Returns a function that takes a MDTraj Trajectory object"""
    contact_function = supported_functions[function_type]
    def obs_function(trajchunk):
        r = md.compute_distances(trajchunk,pairs,periodic=periodic)
        if type(contact_params) == tuple:
            return contact_function(r,*contact_params)
        else:
            return contact_function(r,contact_params)
    return obs_function

def get_covariance_function(obs1,obs2,avgobs1,avgobs2):
    """Returns a function that takes a MDTraj Trajectory object"""
    def obs_function(trajchunk):
        O1 = obs1(trajchunk)
        O2 = obs2(trajchunk)
        O1avg = obs1avg(trajchunk)
        O2avg = obs2avg(trajchunk)
        return np.sum((O1 - O1avg)*(O2 - O2avg))
    return obs_function

def get_edwards_anderson_observable(pairs,function_type,contact_params,periodic=False):
    """ NOT DONE. Returns a function that takes two MDTraj Trajectory objects"""
    contact_function = supported_functions[function_type]
    def obs_function(trajchunk1,trajchunk2):
        r1 = md.compute_distances(trajchunk1,pairs,periodic=periodic)
        r2 = md.compute_distances(trajchunk2,pairs,periodic=periodic)
        cont1 = contact_function(r1,*contact_params)
        cont2 = contact_function(r2,*contact_params)
        #np.dot(cont1,cont2)
        #return np.sum(contact_function(r,*contact_params),axis=1)
    return obs_function

def get_contact_energy_function(pairs,pair_type,eps,contact_params,periodic=False):
    """Returns a function that takes MDTraj Trajectory object and compute pairwise energy """
    from model_builder.models.pairwise_potentials import get_pair_potential
    def obs_function(trajchunk):
        r = md.compute_distances(trajchunk,pairs,periodic=periodic)
        Econtact = np.zeros(trajchunk.n_frames,float)
        for i in range(pairs.shape[0]):
            pair_Vi = get_pair_potential(pair_type[i])
            Econtact += eps[i]*pair_Vi(r[:,i],*contact_params[i])
        return Econtact
    return obs_function


######################################################################
# Contact functions
######################################################################
def tanh_contact(r,r0,widths):
    """Smoothly increasing tanh contact function"""
    return 0.5*(np.tanh(2.*(r0 - r)/widths) + 1)

def weighted_tanh_contact(r,r0,widths,weights):
    """Weighted smoothly increasing tanh contact function"""
    return weights*tanh_contact(r,r0,widths)

def step_contact(r,r0):
    """Step function indicator contact function"""
    return (r <= r0).astype(int)

######################################################################
# Parameterize functions from source data
######################################################################
def get_contact_params(dir,args):
    check_if_supported(args.function)

    n_native_pairs = len(open("%s/native_contacts.ndx" % dir).readlines()) - 1
    if os.path.exists("%s/pairwise_params" % dir):
        r0 = np.loadtxt("%s/pairwise_params" % dir,usecols=(4,),skiprows=1)[1:2*n_native_pairs:2] + 0.1
    elif os.path.exists("%s/native_contact_distances.dat" % dir):
        r0 = np.loadtxt("%s/native_contact_distances.dat" % dir) + 0.1
    else:
        raise IOError("Need source for native contact distances!")
    assert r0.shape[0] == n_native_pairs

    # Get contact function parameters
    if args.function == "w_tanh":
        if (not os.path.exists(args.tanh_weights)) or (args.tanh_weights is None):
            raise IOError("Weights file doesn't exist: %s" % args.tanh_weights)
        else:
            pairs = np.loadtxt(args.tanh_weights,usecols=(0,1),dtype=int)
            widths = args.tanh_scale*np.ones(pairs.shape[0],float)
            weights = np.loadtxt(args.tanh_weights,usecols=(2,),dtype=float)
            contact_params = (r0,widths,weights)
    elif args.function == "tanh":
        pairs = np.loadtxt("%s/native_contacts.ndx" % dir,skiprows=1,dtype=int) - 1
        widths = args.tanh_scale*np.ones(pairs.shape[0],float)
        contact_params = (r0,widths)
    elif args.function == "step":
        pairs = np.loadtxt("%s/native_contacts.ndx" % dir,skiprows=1,dtype=int) - 1
        contact_params = (r0)
    else:
        raise IOError("--function must be in: %s" % supported_functions.keys().__str__())

    return pairs, contact_params


######################################################################
# Functions to loop over trajectories in chunks
######################################################################
def calc_coordinate_for_traj(trajfile,observable_fun,topology,chunksize):
    """Loop over chunks of a trajectory to calculate 1D observable"""
    # In order to save memory we loop over trajectories in chunks.
    obs_traj = []
    for trajchunk in md.iterload(trajfile,top=topology,chunk=chunksize):
        # Calculate observable for trajectory chunk
        obs_traj.extend(observable_fun(trajchunk))
    return np.array(obs_traj)


def calc_coordinate_multiple_trajs(trajfiles,observable_fun,topology,chunksize,save_coord_as=None,collect=True,savepath=None):
    """Loop over directories and calculate 1D observable"""

    obs_all = [] 
    for n in range(len(trajfiles)):
        dir = os.path.dirname(trajfiles[n])
        obs_traj = calc_coordinate_for_traj(trajfiles[n],observable_fun,"%s/%s" % (dir,topology),chunksize)
        if save_coord_as is not None:
            if savepath is not None:
                basename = os.path.basename(dir)
                np.savetxt("%s/%s/%s" % (savepath,basename,save_coord_as),obs_traj)
            else:
                np.savetxt("%s/%s" % (dir,save_coord_as),obs_traj)
        if collect: 
            obs_all.append(obs_traj)
    return obs_all

def bin_covariance_multiple_coordinates_for_traj(trajfile,covar_by_bin,count_by_bin,
        observable1,observable2,obs1_bin_avg,obs2_bin_avg,
        binning_coord,bin_edges,topology,chunksize):
    """Loop over chunks of a trajectory to bin a set of observables along a 1D coordinate"""
    ## TODO test cases:
    # - Two vector-valued observables
    # - One single-valued obesrvable and one-vector-valued observable.
    # - Two single-valued observables

    # In order to save memory we loop over trajectories in chunks.
    start_idx = 0
    for trajchunk in md.iterload(trajfile,top=topology,chunk=chunksize):
        # Calculate observable for trajectory chunk
        obs1_temp = observable1(trajchunk)
        obs2_temp = observable2(trajchunk)
        chunk_size = trajchunk.n_frames
        coord = binning_coord[start_idx:start_idx + chunk_size]
        # Sort frames into bins along binning coordinate.
        for n in range(bin_edges.shape[0]):
            frames_in_this_bin = (coord >= bin_edges[n][0]) & (coord < bin_edges[n][1])
            if frames_in_this_bin.any():
                # Compute the covariance
                delta_obs1 = obs1_temp[frames_in_this_bin] - obs1_bin_avg[n]
                delta_obs2 = obs2_temp[frames_in_this_bin] - obs2_bin_avg[n]
    
                # How should result be collected depending on the number of return values?
                covar_by_bin[n,:,:] = np.dot(delta_obs1.T,delta_obs2)
                count_by_bin[n] += float(sum(frames_in_this_bin))
        start_idx += chunk_size
    return covar_by_bin,count_by_bin

def bin_covariance_multiple_coordinates_for_multiple_trajs(trajfiles,binning_coord,
        observable1,observable2,obs1_bin_avg,obs2_bin_avg,
        n_obs1,n_obs2,bin_edges,topology,chunksize):
    """Compute covariance matrix between two observables over bins"""
    n_bins = bin_edges.shape[0]
    covar_by_bin = np.zeros((n_bins,n_obs1,n_obs_2),float)
    count_by_bin = np.zeros(n_bins,float)
    for n in range(len(trajfiles)):
        dir = os.path.dirname(trajfiles[n])
        covar_by_bin,count_by_bin = bin_covariance_multiple_coordinates_for_traj(trajfiles[n],covar_by_bin,count_by_bin,
                observable1,observable2,obs1_bin_avg,obs2_bin_avg,
                binning_coord[n],bin_edges,"%s/%s" % (dir,topology),chunksize)
    avgcovar_by_bin = (covar_by_bin.T/count_by_bin).T
    return bin_edges,avgcovar_by_bin

def bin_multiple_coordinates_for_traj(trajfile,obs_by_bin,count_by_bin,observable_fun,binning_coord,bin_edges,topology,chunksize):
    """Loop over chunks of a trajectory to bin a set of observables along a 1D coordinate"""
    # In order to save memory we loop over trajectories in chunks.
    start_idx = 0
    for trajchunk in md.iterload(trajfile,top=topology,chunk=chunksize):
        # Calculate observable for trajectory chunk
        obs_temp = observable_fun(trajchunk)
        chunk_size = trajchunk.n_frames
        coord = binning_coord[start_idx:start_idx + chunk_size]
        # Sort frames into bins along binning coordinate. Collect observable average
        for n in range(bin_edges.shape[0]):
            frames_in_this_bin = (coord >= bin_edges[n][0]) & (coord < bin_edges[n][1])
            if frames_in_this_bin.any():
                obs_by_bin[n,:] += np.sum(obs_temp[frames_in_this_bin,:],axis=0)
                count_by_bin[n] += float(sum(frames_in_this_bin))
        start_idx += chunk_size
    return obs_by_bin,count_by_bin


def bin_multiple_coordinates_for_multiple_trajs(trajfiles,binning_coord,observable_function,n_obs,bins,topology,chunksize):
    """Bin multiple coordinates by looping over trajectories"""
    # Calculate pairwise contacts over directories 
    if type(bins) == int:
        n_bins = bins
        counts,bin_edges = np.histogram(np.concatenate(binning_coord),bins=n_bins)
        bin_edges = np.array([[bin_edges[i],bin_edges[i+1]] for i in range(n_bins)])
    else:
        bin_edges = bins
        n_bins = bin_edges.shape[0]
    assert bin_edges.shape[1] == 2
    assert bin_edges.shape[0] == n_bins
    obs_by_bin = np.zeros((n_bins,n_obs),float)
    count_by_bin = np.zeros(n_bins,float)
    for n in range(len(trajfiles)):
        dir = os.path.dirname(trajfiles[n])
        obs_by_bin,count_by_bin = bin_multiple_coordinates_for_traj(trajfiles[n],obs_by_bin,count_by_bin,
                observable_function,binning_coord[n],bin_edges,"%s/%s" % (dir,topology),chunksize)
    avgobs_by_bin = (obs_by_bin.T/count_by_bin).T
    return bin_edges,avgobs_by_bin


global supported_functions
supported_functions = {"step":step_contact,"tanh":tanh_contact,"w_tanh":weighted_tanh_contact}

if __name__ == "__main__":
    pass
