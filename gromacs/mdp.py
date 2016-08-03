
"""
November 11 2013
Purpose:
    Easy way to generate grompp.mdp files for Gromacs simulations. When called
this module returns a grompp.mdp as a string. The requested variables will be 
already set (e.g. temperature, timestep, friction constant).

"""

def constant_energy(nsteps, nstout="1000", dt=0.0001):
    """ Generate grompp.mdp file string. Gromacs 4.5 """
    mdp_string = "; Run control parameters \n"
    mdp_string += "integrator               = md  \n"
    mdp_string += "dt                       = %f \n" % dt
    mdp_string += "nsteps                   = %s \n\n" % nsteps
    mdp_string += "; output control options \n"
    mdp_string += "nstxout                  = 0 \n"
    mdp_string += "nstvout                  = 0 \n"
    mdp_string += "nstfout                  = %s \n" % nstout
    mdp_string += "nstlog                   = 5000 \n"
    mdp_string += "nstenergy                = %s \n" % nstout
    mdp_string += "nstxtcout                = %s \n" % nstout
    mdp_string += "xtc_grps                 = system \n"
    mdp_string += "energygrps               = system \n\n" 
    mdp_string += "; neighborsearching parameters \n"
    mdp_string += "nstlist                  = 20 \n"
    mdp_string += "ns-type                  = grid \n"
    mdp_string += "pbc                      = no \n"
    mdp_string += "periodic_molecules       = no \n"
    mdp_string += "rlist                    = 2.0 \n"
    mdp_string += "rcoulomb                 = 2.0 \n"
    mdp_string += "rvdw                     = 2.0  \n\n"
    mdp_string += "; options for electrostatics and vdw \n"
    mdp_string += "coulombtype              = User \n"
    mdp_string += "vdw-type                 = User \n"
    mdp_string += "table-extension          = 1.0 \n\n"
    mdp_string += "; options for temp coupling \n"
    mdp_string += "Tcoupl                   = no \n"
    mdp_string += "Pcoupl                   = no \n"
    mdp_string += "comm_mode                = angular \n"
    mdp_string += "comm_grps                = System \n"
    return mdp_string

def constant_temperature(T, nsteps, nstout="1000", tau_t=1.):
    """ Generate grompp.mdp file string. Gromacs 4.5 """
    mdp_string = "; Run control parameters \n"
    mdp_string += "integrator               = sd  \n"
    mdp_string += "dt                       = 0.0005 \n"
    mdp_string += "nsteps                   = {} \n\n".format(int(nsteps))
    mdp_string += "; output control options \n"
    mdp_string += "nstxout                  = 0 \n"
    mdp_string += "nstvout                  = 0 \n"
    mdp_string += "nstfout                  = 0 \n"
    mdp_string += "nstlog                   = 5000 \n"
    mdp_string += "nstenergy                = {} \n".format(int(nstout))
    mdp_string += "nstxtcout                = {} \n".format(int(nstout))
    mdp_string += "xtc_grps                 = system \n"
    mdp_string += "energygrps               = system \n\n" 
    mdp_string += "; neighborsearching parameters \n"
    mdp_string += "nstlist                  = 20 \n"
    mdp_string += "ns-type                  = grid \n"
    mdp_string += "pbc                      = no \n"
    mdp_string += "periodic_molecules       = no \n"
    mdp_string += "rlist                    = 2.0 \n"
    mdp_string += "rcoulomb                 = 2.0 \n"
    mdp_string += "rvdw                     = 2.0  \n\n"
    mdp_string += "; options for electrostatics and vdw \n"
    mdp_string += "coulombtype              = User \n"
    mdp_string += "vdw-type                 = User \n"
    mdp_string += "table-extension          = 1.0 \n\n"
    mdp_string += "; options for temp coupling \n"
    mdp_string += "Tcoupl                   = no \n"
    mdp_string += "ld_seed                  = -1 \n"
    mdp_string += "tc-grps                  = system \n"
    mdp_string += "tau_t                    = {} \n".format(tau_t)
    mdp_string += "ref_t                    = {} \n".format(T)
    mdp_string += "Pcoupl                   = no \n\n"
    mdp_string += "; generate velocities for startup run \n"
    mdp_string += "gen_vel                  = yes \n"
    mdp_string += "gen_temp                 = {} \n".format(T)
    mdp_string += "gen_seed                 = -1 \n\n"
    mdp_string += "; remove center of mass\n"
    mdp_string += "comm_mode                = angular \n"
    mdp_string += "comm_grps                = System \n"
    return mdp_string

def simulated_annealing(Tlist,pslist,nsteps,nstout="1000"):
    """ Generate grompp.mdp file string. Gromacs 4.6 """
    # Should we assert that the total number of steps is longer than the
    # annealing schedule? 

    annealtimes = ""
    annealtemps = ""
    for i in range(len(Tlist)):
        annealtimes += "%.2f " % (pslist[i] + sum(pslist[:i]))
        annealtemps += "%.2f " % Tlist[i]

    mdp_string = "; Run control parameters \n"
    mdp_string += "integrator               = sd  \n"
    mdp_string += "dt                       = 0.0005 \n"
    mdp_string += "nsteps                   = %s \n\n" % nsteps
    mdp_string += "; output control options \n"
    mdp_string += "nstxout                  = 0 \n"
    mdp_string += "nstvout                  = 0 \n"
    mdp_string += "nstfout                  = 0 \n"
    mdp_string += "nstlog                   = 5000 \n"
    mdp_string += "nstenergy                = %s \n" % nstout
    mdp_string += "nstxtcout                = %s \n" % nstout
    mdp_string += "xtc_grps                 = system \n"
    mdp_string += "energygrps               = system \n\n" 
    mdp_string += "; neighborsearching parameters \n"
    mdp_string += "nstlist                  = 20 \n"
    mdp_string += "ns-type                  = grid \n"
    mdp_string += "pbc                      = no \n"
    mdp_string += "periodic_molecules       = no \n"
    mdp_string += "rlist                    = 2.0 \n"
    mdp_string += "rcoulomb                 = 2.0 \n"
    mdp_string += "rvdw                     = 2.0  \n\n"
    mdp_string += "; options for electrostatics and vdw \n"
    mdp_string += "coulombtype              = User \n"
    mdp_string += "vdw-type                 = User \n"
    mdp_string += "table-extension          = 1.0 \n\n"
    mdp_string += "; options for temp coupling \n"
    mdp_string += "Tcoupl                   = no \n"
    mdp_string += "ld_seed                  = -1 \n"
    mdp_string += "tc-grps                  = system \n"
    mdp_string += "tau_t                    = 1 \n"
    mdp_string += "ref_t                    = %s \n" % Tlist[0]
    mdp_string += "Pcoupl                   = no \n\n"
    mdp_string += "; generate velocities for startup run \n"
    mdp_string += "gen_vel                  = yes \n"
    mdp_string += "gen_temp                 = %s \n" % Tlist[0]
    mdp_string += "gen_seed                 = -1 \n\n"
    mdp_string += "; remove center of mass\n"
    mdp_string += "comm_mode                = angular \n"
    mdp_string += "comm_grps                = System \n"
    mdp_string += "; simulated annealing schedule\n"
    mdp_string += "annealing                = single\n"
    mdp_string += "annealing-npoints        = %d\n" % len(Tlist)
    mdp_string += "annealing-time           = %s\n" % annealtimes
    mdp_string += "annealing-temp           = %s\n" % annealtemps
    return mdp_string

def energy_minimization(integrator="l-bfgs", etol=1.):
    assert integrator in ["l-bfgs"]

    mdp_string =   "integrator  = %s\n" % integrator
    mdp_string +=  "emtol       = %.1f\n" % etol
    mdp_string +=  "emstep      = 0.005\n"
    mdp_string +=  "nsteps      = 50000\n"
    mdp_string +=  "; neighborsearching parameters\n"
    mdp_string +=  "nstlist                  = 10\n"
    mdp_string +=  "ns-type                  = grid\n"
    mdp_string +=  "pbc                      = no\n"
    mdp_string +=  "periodic_molecules       = no\n"
    mdp_string +=  "rlist                    = 2.0\n"
    mdp_string +=  "rcoulomb                 = 2.0\n"
    mdp_string +=  "rvdw                     = 2.0\n"
    mdp_string +=  "; options for electrostatics and vdw\n"
    mdp_string +=  "coulombtype              = User\n"
    mdp_string +=  "vdw-type                 = User\n"
    mdp_string +=  "table-extension          = 1.0\n"
    return mdp_string


def normal_modes(etol=1.):
    mdp_string =   "integrator  = nm\n"
    mdp_string +=  "emtol       = %.1f\n" % etol
    mdp_string +=  "emstep      = 0.005\n"
    mdp_string +=  "nsteps      = 50000\n"
    mdp_string +=  "; neighborsearching parameters\n"
    mdp_string +=  "nstlist                  = 10\n"
    mdp_string +=  "ns-type                  = grid\n"
    mdp_string +=  "pbc                      = no\n"
    mdp_string +=  "periodic_molecules       = no\n"
    mdp_string +=  "rlist                    = 2.0\n"
    mdp_string +=  "rcoulomb                 = 2.0\n"
    mdp_string +=  "rvdw                     = 2.0\n"
    mdp_string +=  "; options for electrostatics and vdw\n"
    mdp_string +=  "coulombtype              = User\n"
    mdp_string +=  "vdw-type                 = User\n"
    mdp_string +=  "table-extension          = 1.0\n"
    return mdp_string

#def pull_code(group1_name, group2_name, kumb, r0):
#    """ GROMACS 5."""
#    mdp_string =   "; Pull code             \n"
#    mdp_string +=  "pull                    = yes       \n"
#    mdp_string +=  "pull_ngroups            = 2\n"
#    mdp_string +=  "pull_ncoords            = 1\n"
#    mdp_string +=  "pull_group1_name        = {}\n".format(group1_name)
#    mdp_string +=  "pull_group2_name        = {}\n".format(group2_name)
#    mdp_string +=  "pull_coord1_type        = umbrella      ; harmonic biasing force\n"
#    mdp_string +=  "pull_coord1_geometry    = distance      ; simple distance increase\n"
#    mdp_string +=  "pull_coord1_groups      = 1 2\n"
#    mdp_string +=  "pull_coord1_dim         = Y Y Y\n"
#    mdp_string +=  "pull_coord1_rate        = 0.0\n"
#    mdp_string +=  "pull_coord1_init        = {:.8f}\n".format(r0)
#    mdp_string +=  "pull_coord1_k           = {:.8f}          ; kJ mol^-1 nm^-2\n".format(kumb)
#    #mdp_string +=  "pull_coord1_start       = yes           ; define initial COM distance > 0\n"
#    return mdp_string

def pull_code(group1_name, group2_name, kumb, r0):
    """ GROMACS 4.5.6"""
    mdp_string =   "; Pull code             \n"
    mdp_string +=  "pull                    = umbrella       \n"
    mdp_string +=  "pull_geometry           = distance      ; simple distance increase\n"
    mdp_string +=  "pull_dim                = Y Y Y\n"
    mdp_string +=  "pull_ngroups            = 1\n"
    mdp_string +=  "pull_group0             = {}\n".format(group1_name)
    mdp_string +=  "pull_group1             = {}\n".format(group2_name)
    mdp_string +=  "pull_init1              = {:.8f}\n".format(r0)
    mdp_string +=  "pull_rate1              = 0.0\n"
    mdp_string +=  "pull_k1                 = {:.8f}          ; kJ mol^-1 nm^-2\n".format(kumb)
    #mdp_string +=  "pull_start       = yes           ; define initial COM distance > 0\n"
    return mdp_string
