import os
import numpy as np
import pickle

import mdtraj as md
import pyemma.thermo as thermo
import pyemma.coordinates as coor
import pyemma.msm as msm
import pyemma.plots as mplt
from pyemma.coordinates.data.featurization.misc import CustomFeature


global KB
KB = 0.0083145

class DummyTram(object):
    def __init__(self):
        pass

class SbmEnergyFeature(CustomFeature):
    """Create a feature for Time Independent Component Analysis

    Description
    -----------
    Playing around with the idea that the terms of the potential energy are
    good to cluster configurations. Turns out they are not great because they
    map configurations on top of one another.

    Examples
    --------
    >>> import model_builder as mdb
    >>> model = mdb.inputs.load_model(name + ".ini")[0]
    >>> EnergyFeature = SbmEnergyFeature(model, n_native=n_native_pairs)

    """

    def __init__(self, model, n_native=None):

        if not (n_native is None):
            self.Vdih = [ x.V for x in model.Hamiltonian._dihedrals ]
            self.Vpair = [ x.V for x in model.Hamiltonian._pairs ]
            self.dihedrals = model.Hamiltonian._dihedral_idxs
            self.pairs = model.Hamiltonian._pair_idxs
        else:
            self.Vdih = [ x.V for x in model.Hamiltonian._dihedrals ]
            self.Vpair = [ x.V for x in model.Hamiltonian._pairs[:n_native] ]
            self.dihedrals = model.Hamiltonian._dihedral_idxs
            self.pairs = model.Hamiltonian._pair_idxs[:n_native,:]
        self.dimension = len(self.dihedrals) + len(self.pairs)

    def dimension(self):
        return self.dimension

    def transform(self, traj):

        # compute dihedral energy
        phi = md.compute_dihedrals(traj, self.dihedrals)
        Edih = np.array(map(lambda x,y: x(y), self.Vdih, phi.T)).T

        # compute pair energy
        r = md.compute_distances(traj, self.pairs)
        Epair = np.array(map(lambda x,y: x(y), self.Vpair, r.T)).T

        return np.hstack((Edih, Epair))

def get_organized_temps(tempfile=None, temperature_dirs=None):
    """Get directory names by temperature"""
    if not tempfile is None:
        with open(tempfile, "r") as fin:
            temperature_dirs = fin.read().split()
    else:
        if temperature_dirs is None:
            raise IOError("need to input tempfile or temperature_dirs")

    organized_temps = {}
    for i in range(len(temperature_dirs)):
        temp_dir = temperature_dirs[i]
        temp_T = float((temp_dir.split("/")[0]).split("_")[1])
        if not temp_T in organized_temps.keys():
            organized_temps[temp_T] = [temp_dir]
        else:
            organized_temps[temp_T].append(temp_dir)

    return organized_temps

def get_ca_pair_idxs(top, pair_skip=5):

    ca_idxs = top.select("name CA")[::pair_skip]
    ca_pair_idxs = np.array([ [ca_idxs[i], ca_idxs[j]] for i in range(len(ca_idxs)) for j in range(i + 1, len(ca_idxs)) ])
    return ca_pair_idxs

def get_awsem_phi_psi_idxs(top):

    phi_specs = lambda res: [[res.index - 1, "C"], [res.index, "N"], [res.index, "CA"], [res.index, "C"]]
    psi_specs = lambda res: [[res.index, "N"], [res.index, "CA"], [res.index, "C"], [res.index + 1, "N"]]
    select_atm = lambda idx, name: top.select("resid {} and name {}".format(idx, name))[0]

    # Dihedral angles
    # C_i-1 N_i CA_i C_i
    # N_i CA_i C_i N_i+1
    dih_idxs = []
    for chain in top.chains:
        # We skip terminal residues. They are missing N or C atoms.
        for i in range(1, chain.n_residues - 1):
            res = chain.residue(i)
            res_phi_idxs = [ select_atm(idx, name) for idx, name in phi_specs(res) ]
            res_psi_idxs = [ select_atm(idx, name) for idx, name in psi_specs(res) ]

            # add to regular dihedrals
            dih_idxs.append(res_phi_idxs)
            dih_idxs.append(res_psi_idxs)

    return np.array(dih_idxs)

######################################################################
# Functions to help build Multi-Ensemble Markov Models (MEMM)
######################################################################

def tanh_contact(traj, pairs, r0, widths):
    r = md.compute_distances(traj, pairs)
    return 0.5*(np.tanh((r0 - r)/widths) + 1)

def dih_cosine(traj, dih_idxs, phi0):
    phi = md.compute_dihedrals(traj, dih_idxs)
    return np.cos(phi - phi0)

def default_ca_sbm_features(feat, topfile, pairsfile=None):
    """Default features for a C-alpha structure-based model (SBM) are dihedral angles and """
    ref = md.load(topfile)
    dihedral_idxs = np.array([ [i, i + 1, i + 2, i + 3] for i in range(ref.n_residues - 3) ])
    if pairsfile is None:
        # add all pairs
        pair_idxs = np.loadtxt(pairsfile, dtype=int) - 1
    else:
        # add pairs from file
        pair_idxs = np.loadtxt(pairsfile, dtype=int) - 1

    # cluster trajectories
    feat.add_distances(pair_idxs)
    feat.add_dihedrals(dihedral_idxs)

    return feat

def sbm_contact_features(feat, pairwise_file, n_native_pairs, skip_nn=10, native_only=False, nonnative_only=False):
    """Contact feature using tanh switching function"""

    pair_idxs = np.loadtxt(pairwise_file, usecols=(0,1), dtype=int) - 1
    r0 = np.loadtxt(pairwise_file, usecols=(5,), dtype=np.float32) + 0.1
    widths = np.loadtxt(pairwise_file, usecols=(6,), dtype=np.float32)

    if native_only:
        pair_idxs = pair_idxs[:n_native_pairs,:]
        r0 = r0[:n_native_pairs]
        widths = widths[:n_native_pairs]
    else:
        if nonnative_only:
            pair_idxs = pair_idxs[n_native_pairs::skip_nn,:]
            r0 = r0[n_native_pairs::skip_nn]
            widths = widths[n_native_pairs::skip_nn]
        else:
            pair_idxs = np.vstack((pair_idxs[:n_native_pairs,:], pair_idxs[n_native_pairs::skip_nn,:]))
            r0 = np.concatenate((r0[:n_native_pairs], r0[n_native_pairs::skip_nn]))
            widths = np.concatenate((widths[:n_native_pairs], widths[n_native_pairs::skip_nn]))

    feat.add_custom_feature(CustomFeature(tanh_contact, pair_idxs, r0, widths, dim=len(pair_idxs)))
    #feat.add_custom_feature(CustomFeature(tanh_contact, pair_idxs, r0, widths))

    feature_info = {'pairs':pair_idxs, 'r0':r0, 'widths':widths, 'dim':len(pair_idxs)}

    return feat, feature_info 


def default_awsem_features(feat, pair_skip=5):
    """Adds features that appear to work well for awsem
    
    Parameters
    ----------
    feat : obj, pyemma.coordinates.featurizer
        Featurizer to be used for markov state modeling.
    pair_skip : int, opt.
        Number of atoms to skip when constructing set of pairwise distances.
    """
    dih_idxs = get_awsem_phi_psi_idxs(feat.topology)
    pair_idxs = get_ca_pair_idxs(feat.topology, pair_skip=pair_skip)
    feat.add_dihedrals(dih_idxs)
    feat.add_distances(pair_idxs)
    return feat


def multi_temperature_tram(feat, trajfiles, temperatures, dtrajs=None, stride=1, tica_lag=100,
        keep_tica_dims=20, n_clusters=100, tram_lag=100, engfile="Etot.dat", usecols=(1,), kb=0.0083145):
    """
    Parameters
    ----------
    feat : obj, pyemma.coor.featurizer
        Featurizer object that already has the appropriate features added.
    trajfiles : list
        Names of trajectories to include in estimation.
    temperatures : list
        Temperatures of corresponding trajectories.
    stride : int
        Number of frames to skip in tica and clustering.
    tica_lag : int
        Lagtime to use for constructing tica.
    keep_tica_dims : int
        Number of dimensions to keep from tica. Somewhat ambiguous.
    n_clusters : int
        Number of clusters for kmeans. Somewhat ambiguous. 
    """

    dirs = [ os.path.dirname(x) for x in trajfiles ]
    beta = [ 1./(kb*x) for x in temperatures ]

    if dtrajs is None:
        inp = coor.source(trajfiles, feat)

        tica_obj = coor.tica(inp, lag=tica_lag, dim=keep_tica_dims, stride=stride)
        Y = tica_obj.get_output()

        cl = coor.cluster_kmeans(data=Y, k=n_clusters, stride=stride)
        dtrajs = cl.dtrajs

    # dimensionless energy
    if engfile.endswith("npy"):
        energy_trajs = [ beta[i]*np.load("{}/{}".format(dirs[i], engfile)) for i in range(len(dirs)) ]
    else:
        energy_trajs = [ beta[i]*np.loadtxt("{}/{}".format(dirs[i], engfile), usecols=usecols) for i in range(len(dirs)) ]
    temp_trajs = [ kb*temperatures[i]*np.ones(energy_trajs[i].shape[0], float) for i in range(len(dirs)) ]

    # dTRAM approach
    tram = thermo.estimate_multi_temperature(energy_trajs, temp_trajs,
            dtrajs, energy_unit='kT', temp_unit='kT', estimator='tram',
            lag=tram_lag, maxiter=2000000, maxerr=1e-10)

    return dirs, dtrajs, tram

def save_multi_temperature_tram(dirs, dtrajs, tram):
    """Saves MEMM

    Parameters
    ----------
    dirs : list
        List of directory names.
    dtrajs : list of np.ndarray
        List of discrete trajectories corresponding to trajectories held in dirs.
    tram : obj, pyemma.thermotools.dTRAM
        A TRAM estimator object from the pyemma software package.

    """
    # save information needed to rebuild the MSM's created by dTRAM.
    if not os.path.exists("tram"):
        os.mkdir("tram")
    os.chdir("tram")

    dtraj_info = { dirs[x]:dtrajs[x] for x in range(len(dirs)) }
    dtraj_info["dirs"] = dirs
    with open("dtrajs.pkl", "wb") as fhandle:
        pickle.dump(dtraj_info, fhandle)

    tram_info = {}
    tram_info["temperatures"] = tram.temperatures/KB
    tram_info["tram_f"] = tram.f
    tram_info["tram_f_therm"] = tram.f_therm

    for k in range(tram.nthermo):
        temperature = tram.temperatures[k]/KB
        tram_info["{:.2f}_active_set".format(temperature)] = tram.model_active_set[k]
        tram_info["{:.2f}_stationary_distribution".format(temperature)] = tram.models[k].stationary_distribution
        tram_info["{:.2f}_transition_matrix".format(temperature)] = tram.models[k].transition_matrix

    with open("tram.pkl", "wb") as fhandle:
        pickle.dump(tram_info, fhandle)

    os.chdir("..")

def save_markov_state_models(T, models):

    msm_info = {}
    msm_info["temperature"] = T
    msm_info["lagtimes"] = [ x.lagtime for x in models ]

    for i in range(len(models)):
        lagtime = models[i].lagtime
        msm_info[str(lagtime)] = models[i].transition_matrix

    with open("msm.pkl", "wb") as fhandle:
        pickle.dump(msm_info, fhandle)

def load_markov_state_models():

    os.chdir("msm")
    with open("dtrajs.pkl", "rb") as fhandle:
        dtrajs_info = pickle.load(fhandle)

    dirs = dtrajs_info["dirs"]
    dtrajs = [ dtrajs_info[x] for x in dirs ]

    with open("msm.pkl", "rb") as fhandle:
        msm_info = pickle.load(fhandle)
    lagtimes = msm_info["lagtimes"]

    models = []
    for i in range(len(lagtimes)):
        models.append(msm.markov_model(msm_info[str(lagtimes[i])]))

    os.chdir("..")

    return dirs, dtrajs, lagtimes, models

def load_multi_temperature_tram():
    """Loads a MEMM

    Returns
    -------
    dirs : list
        List of directory names.
    dtrajs : list of np.ndarray
        List of discrete trajectories corresponding to trajectories held in dirs.
    tram : obj, pyemma.thermotools.dTRAM
        A dTRAM estimator object from the pyemma software package.

    """
    os.chdir("tram")
    with open("dtrajs.pkl", "rb") as fhandle:
        dtraj_pkl = pickle.load(fhandle)
        dirs = dtraj_pkl["dirs"]
        dtrajs = [ dtraj_pkl[x] for x in dirs ]

    tram = DummyTram()
    with open("tram.pkl", "rb") as fhandle:
        tram_pkl = pickle.load(fhandle)
        tram.temperatures = tram_pkl["temperatures"]*KB
        tram.f = tram_pkl["tram_f"]
        tram.f_therm = tram_pkl["tram_f_therm"]
        tram.ntherm = len(tram.temperatures)

        tram.model_active_set = [ tram_pkl["{:.2f}_active_set".format(x)] for x in tram.temperatures/KB ]
        model_pi = [ tram_pkl["{:.2f}_stationary_distribution".format(x)] for x in tram.temperatures/KB ]
        model_T = [ tram_pkl["{:.2f}_transition_matrix".format(x)] for x in tram.temperatures/KB ]
        tram.models = [ msm.MSM(model_T[x], pi=model_pi[x]) for x in range(tram.ntherm) ]

    os.chdir("..")
    return dirs, dtrajs, tram

def sort_observable_into_clusters(clust_idxs, obs_trajs, dtrajs):
    clust_obs = [ None for x in range(len(clust_idxs)) ]

    # get data from all directories at this thermo state.
    for n in range(len(obs_trajs)):
        q = obs_trajs[n]
        dtraj = dtrajs[n]
        for i in range(len(clust_idxs)):
            # collect the frames that are in each cluster
            obs_in_clust = (dtraj == clust_idxs[i])
            if np.any(obs_in_clust):
                if clust_obs[i] is None:
                    clust_obs[i] = q[obs_in_clust]
                else:
                    clust_obs[i] = np.concatenate((clust_obs[i], q[obs_in_clust]))

    return clust_obs

