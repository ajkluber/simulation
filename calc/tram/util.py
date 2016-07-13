import os
import numpy as np
import pickle

import mdtraj as md
import pyemma.thermo as thermo
import pyemma.coordinates as coor
import pyemma.msm as msm
import pyemma.plots as mplt

global KB
KB = 0.0083145

class DummyTram(object):
    def __init__(self):
        pass

def get_organized_temps(tempfile):
    """Get directory names by temperature"""
    with open(tempfile, "r") as fin:
        temperature_dirs = fin.read().split()

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
        A dTRAM estimator object from the pyemma software package.

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
