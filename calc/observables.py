import os
import numpy as np

import mdtraj

from model_builder.models.pairwise_potentials import get_pair_potential

# Should model code after pyemma and mdtraj packages.

def get_observable_function(args):
    supported_contact_functions = {"step":step_contact,"tanh":tanh_contact,"w_tanh":weighted_tanh_contact}

def _describe_atom(topology, index):
    """
    Returns a string describing the given atom

    :param topology:
    :param index:
    :return:
    """
    traj = mdtraj.load(topology)
    at = traj.topology.atom(index)
    return "%s %i %s %i" % (at.residue.name, at.residue.index, at.name, at.index)

class Contacts(object):
    def __init__(self, top, pairs, r0):
        self.prefix_label = "CONTACT"
        self.top = top
        self.pairs = pairs
        self.r0 = r0

    def describe(self):
        labels = ["%s %s - %s" % (self.prefix_label,
                                  _describe_atom(self.top, pair[0]),
                                  _describe_atom(self.top, pair[1]))
                  for pair in self.pairs]
        return labels

class TanhContacts(Contacts):
    """Smoothly increasing tanh contact function"""

    def __init__(self, top, pairs, r0, widths, periodic=False):
        Contacts.__init__(self, top, pairs, r0)
        self.prefix_label = "TANHCONTACT"
        self.widths = widths
        self.periodic = periodic
        self.dimension = pairs.shape[0]
        self.symbol = "$Q_{tanh,i}$"

    def map(self, traj):
        r = mdtraj.compute_distances(traj, self.pairs, periodic=self.periodic)
        return 0.5*(np.tanh(2.*(self.r0 - r)/self.widths) + 1)

class TanhContactSum(TanhContacts):
    """Sum of tanh contacts"""

    def __init__(self, top, pairs, r0, widths, periodic=False):
        TanhContacts.__init__(self, top, pairs, r0, widths, periodic=periodic)
        self.prefix_label = "TANHCONTACTSUM"
        self.dimension = 1
        self.symbol = "$Q_{tanh}$"

    def map(self, traj):
        r = mdtraj.compute_distances(traj, self.pairs, periodic=self.periodic)
        return np.sum(0.5*(np.tanh(2.*(self.r0 - r)/self.widths) + 1),axis=1)

class WeightedTanhContacts(TanhContacts):
    """Weighted tanh contacts"""

    def __init__(self, top, pairs, r0, widths, weights, periodic=False):
        TanhContacts.__init__(self, top, pairs, r0, widths, periodic=periodic)
        self.prefix_label = "WTANHCONTACT"
        self.weights = weights
        self.dimension = pairs.shape[0]
        self.symbol = "$Q_{tanh,i}^{w}$"

    def map(self, traj):
        r = mdtraj.compute_distances(traj, self.pairs, periodic=self.periodic)
        return self.weights*0.5*(np.tanh(2.*(self.r0 - r)/self.widths) + 1)

class WeightedTanhContactSum(WeightedTanhContacts):
    """Sum of weighted tanh contacts"""

    def __init__(self, top, pairs, r0, widths, weights, periodic=False):
        WeightedTanhContacts.__init__(self, top, pairs, r0, widths, weights, periodic=periodic)
        self.prefix_label = "WTANHCONTACTSUM"
        self.dimension = 1
        self.symbol = "$Q_{tanh}^{w}$"

    def map(self, traj):
        r = mdtraj.compute_distances(traj, self.pairs, periodic=self.periodic)
        return np.sum(self.weights*0.5*(np.tanh(2.*(self.r0 - r)/self.widths) + 1),axis=1)

class StepContacts(Contacts):
    """Step function contacts"""

    def __init__(self, top, pairs, r0, periodic=False):
        Contacts.__init__(self, top, pairs, r0)
        self.periodic = periodic
        self.prefix_label = "STEPCONTACT"
        self.dimension = pairs.shape[0]
        self.symbol = "$Q_{step,i}$"   

    def map(self,traj):
        r = mdtraj.compute_distances(traj, self.pairs, periodic=self.periodic)
        return (r <= self.r0).astype(int)

class StepContactSum(StepContacts):
    """Sum of step function contacts"""

    def __init__(self, top, pairs, r0, periodic=False):
        StepContacts.__init__(self, top, pairs, r0, periodic=periodic)
        self.prefix_label = "STEPCONTACTSUM"
        self.dimension = 1
        self.symbol = "$Q_{step}$"   

    def map(self,traj):
        r = mdtraj.compute_distances(traj, self.pairs, periodic=self.periodic)
        return np.sum((r <= self.r0).astype(int),axis=1)

class PairEnergy(object):
    """Pairwise energy"""
    def __init__(self, top, pairs, pair_type, eps, pair_params, periodic=False):
        self.top = top
        self.pairs = pairs
        self.pair_type = pair_type
        self.eps = eps
        self.pair_params = pair_params
        self.dimension = pairs.shape[0]
        self.periodic = periodic
        self.symbol = "$E_i$"
    
    def map(self,traj):
        r = mdtraj.compute_distances(traj, self.pairs, periodic=self.periodic)
        Epair = np.zeros((traj.n_frames, self.dimension), float)
        for i in range(self.dimension):
            Vi = get_pair_potential(self.pair_type[i])
            Epair[:,i] = self.eps[i]*Vi(r[:,i],*self.pair_params[i])
        return Epair

class PairEnergySum(PairEnergy):
    """Sum of pairwise energy"""
    def __init__(self, top, pairs, pair_type, eps, pair_params, periodic=False):
        PairEnergy.__init__(self, top, pairs, pair_type, eps, pair_params, periodic=periodic)
        self.dimension = 1
        self.symbol = "$E$"
    
    def map(self,traj):
        r = mdtraj.compute_distances(traj, self.pairs, periodic=self.periodic)
        Epair = np.zeros((traj.n_frames, self.dimension), float)
        for i in range(self.pairs.shape[0]):
            Vi = get_pair_potential(self.pair_type[i])
            Epair[:,0] += self.eps[i]*Vi(r[:,i],*self.pair_params[i])
        return Epair

def calculate_observable(trajfiles, observable, chunksize=1000, collect=True, saveas=None, savepath=None):
    """Calculate observable over trajectories"""

    obs_all = [] 
    for n in range(len(trajfiles)):
        obs_traj = []
        for trajchunk in mdtraj.iterload(trajfiles[n],top=observable.top,chunk=chunksize):
            obs_traj.extend(observable.map(trajchunk))
        obs_traj = np.array(obs_traj)

        if collect: 
            obs_all.append(obs_traj)

        if saveas is not None:
            dir = os.path.dirname(trajfiles[n])
            if savepath is not None:
                # Save observable in another directory
                basename = os.path.basename(dir)
                np.savetxt("%s/%s/%s" % (savepath,basename,saveas),obs_traj)
            else:
                # Save observable next to trajectory 
                np.savetxt("%s/%s" % (dir,saveas),obs_traj)
    return obs_all

def bin_observable(trajfiles, observable, binning_coord, bin_edges, chunksize=10000):
    """Bin observable over trajectories"""

    assert len(binning_coord[0].shape) == 1
    assert bin_edges.shape[1] == 2

    obs_by_bin = np.zeros((bin_edges.shape[0],observable.dimension),float)
    count_by_bin = np.zeros(bin_edges.shape[0],float)
    for i in range(len(trajfiles)):
        start_idx = 0
        for trajchunk in mdtraj.iterload(trajfiles[i],top=observable.top,chunk=chunksize):
            obs_temp = observable.map(trajchunk)
            chunk_size = trajchunk.n_frames
            coord = binning_coord[i][start_idx:start_idx + chunk_size]
            for n in range(bin_edges.shape[0]):
                frames_in_this_bin = (coord >= bin_edges[n][0]) & (coord < bin_edges[n][1])
                if np.any(frames_in_this_bin):
                    obs_by_bin[n,:] += np.sum(obs_temp[frames_in_this_bin],axis=0)
                    count_by_bin[n] += float(sum(frames_in_this_bin))
            start_idx += chunk_size
            
    obs_bin_avg = (obs_by_bin.T/count_by_bin).T
    return obs_bin_avg

def bin_observable_covariance(trajfiles, observable1, observable2, obs1_bin_avg, obs2_bin_avg, 
            binning_coord, bin_edges, chunksize=10000):
    """Covariance bin observable over trajectories"""

    assert len(binning_coord[0].shape) == 1
    assert bin_edges.shape[1] == 2

    covar_bin_avg = np.zeros((bin_edges.shape[0],observable1.dimension,observable2.dimension),float)
    count_by_bin = np.zeros(bin_edges.shape[0],float)
    counter = 0 
    for i in range(len(trajfiles)):
        start_idx = 0
        for trajchunk in mdtraj.iterload(trajfiles[i],top=observable1.top,chunk=chunksize):
            obs1_temp = observable1.map(trajchunk)
            obs2_temp = observable2.map(trajchunk)
            chunk_size = trajchunk.n_frames
            coord = binning_coord[i][start_idx:start_idx + chunk_size]
            for n in range(bin_edges.shape[0]):
                frames_in_this_bin = (coord >= bin_edges[n][0]) & (coord < bin_edges[n][1])
                if np.any(frames_in_this_bin):
                    covar_bin_avg[n,:,:] += np.dot((obs1_temp[frames_in_this_bin] - obs1_bin_avg[n]).T,(obs2_temp[frames_in_this_bin] - obs2_bin_avg[n]))
                    count_by_bin[n] += float(sum(frames_in_this_bin))
            start_idx += chunk_size
            print "chunk =  %d " % counter 
            counter += 1
    for n in range(bin_edges.shape[0]):
        if count_by_bin[n] > 0:
            covar_bin_avg[n,:,:] /= count_by_bin[n]
    return covar_bin_avg


if __name__ == "__main__":
    trajfiles = [ x.rstrip("\n") for x in open("ticatrajs","r").readlines() ]
    dir = os.path.dirname(trajfiles[0])

    # parameterize contact function
    pairs = np.loadtxt("%s/native_contacts.ndx" % dir, skiprows=1, dtype=int) - 1
    n_native_pairs = pairs.shape[0]
    r0 = np.loadtxt("%s/pairwise_params" % dir, usecols=(4,))[1:2*n_native_pairs:2]
    top = "%s/Native.pdb" % dir
    widths = 0.05*np.ones(n_native_pairs, float)

    ########################################
    # TESTING CONTACT OBSERVABLES
    ########################################
#    r0_cont = r0 + 0.1
#    qtanhsum_obs = TanhContactSum(top, pairs, r0_cont, widths)
#    #qstep_obs = StepContactSum(top, pairs, r0_cont)
#    #qstep_obs = StepContacts(top, pairs, r0_cont)
#    qtanh_obs = TanhContacts(top, pairs, r0_cont, widths)
#
#    #qtanh = calculate_observable(trajfiles, qtanhsum_obs)
    qtanh = [ np.loadtxt("%s/Qtanh_0_05.dat" % os.path.dirname(x)) for x in trajfiles ]
#
    n, bins= np.histogram(qtanh,bins=30)
    bin_edges = np.array([ [bins[i], bins[i+1]] for i in range(len(bins) - 1 ) ])
#    
#    # Works!
#    #qstep_bin_avg = bin_observable(trajfiles, qtanh_obs, qtanh, bin_edges)
#    qtanh_bin_avg = bin_observable(trajfiles, qtanh_obs, qtanh, bin_edges)


    ########################################
    # TESTING PAIRWISE ENERGY OBSERVABLES
    ########################################

    # parameterize pair energy observable
    pair_type = np.loadtxt("%s/pairwise_params" % dir, usecols=(3,), dtype=int)[1:2*n_native_pairs:2]
    eps = np.loadtxt("%s/model_params" % dir)[1:2*n_native_pairs:2]

    pair_params = []
    for line in open("%s/pairwise_params" % dir,"r"):
        if line.startswith("#"):
            continue
        else:
            pair_params.append(tuple([ float(x) for x in line.split()[4:] ]))
    nat_pair_params = pair_params[1:2*n_native_pairs:2]

    #def __init__(self, top, pairs, pair_type, eps, pair_params, periodic=False):
    paireng_obs = PairEnergy(top, pairs, pair_type, eps, nat_pair_params)
    pairengsum_obs = PairEnergySum(top, pairs, pair_type, eps, nat_pair_params)
    #Enat = calculate_observable(trajfiles, pairengsum_obs)
    Enatpair_bin_avg = bin_observable(trajfiles, paireng_obs, qtanh, bin_edges)
    #Enat_bin_avg = bin_observable(trajfiles, pairengsum_obs, qtanh, bin_edges)

    #dEnat_bin_avg = bin_observable_covariance(trajfiles, pairengsum_obs, pairengsum_obs, Enat_bin_avg, Enat_bin_avg, qtanh, bin_edges)
    #dEnat2_bin_avg = bin_observable_covariance(trajfiles, pairengsum_obs, pairengsum_obs, Enat_bin_avg, Enat_bin_avg, qtanh, bin_edges)
    dEnatpair2_bin_avg = bin_observable_covariance(trajfiles, paireng_obs, paireng_obs, Enatpair_bin_avg, Enatpair_bin_avg, qtanh, bin_edges)













