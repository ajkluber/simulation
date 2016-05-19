import numpy as np

def header_string(boundary):
    """LAMMPS header"""

    header_string = "# LAMMPS simulation script\n\n"

    header_string += "units real\n"
    header_string += "timestep 2\n"
    header_string += "dimension   3\n\n"

    # Set boundary conditions and neighborlist settings
    header_string += "boundary {} {} {}\n".format(boundary[0],boundary[1],boundary[2])
    header_string += "neighbor    10 bin\n"
    header_string += "neigh_modify    delay 5\n"

    # Turn off sorting
    header_string += "# AWSEM custom atom type and function\n"
    header_string += "atom_modify sort 0 0.0\n"
    header_string += "special_bonds fene\n"
    header_string += "atom_style peptide\n"
    header_string += "bond_style harmonic\n"
    header_string += "pair_style vexcluded 2 3.5 3.5\n\n"
    return header_string

def data_and_pair_coeff_string(topfile):

    string = "# Read in the molecular topology\n"
    string += "read_data {}\n\n".format(topfile)

    string += "# Set the excluded volume of different pairs of atom types?\n"
    string += "pair_coeff * * 0.0\n"
    string += "pair_coeff 1 1 20.0 3.5 4.5\n"
    string += "pair_coeff 1 4 20.0 3.5 4.5\n"
    string += "pair_coeff 4 4 20.0 3.5 4.5\n"
    string += "pair_coeff 3 3 20.0 3.5 3.5\n\n"
    return string 

def group_def_string(CA_idxs, CB_HB_idxs, O_idxs, extra_group_defs):

    string = "# Define atom groups\n"
    string += "group alpha_carbons id {}\n".format(" ".join([ str(x) for x in CA_idxs ]))
    string += "group beta_atoms id {}\n".format(" ".join([ str(x) for x in CB_HB_idxs ]))
    string += "group oxygens id {}\n\n".format(" ".join([ str(x) for x in O_idxs ]))

    string += "# Extra group definitions.\n"
    string += extra_group_defs

    return string

def fix_langevin_integrator(T, damping_const):
    """fix ID group-ID langevin Tstart Tstop damp seed keyword values"""

    string = "# Set integrator, temperature, damping constant.\n"
    string += "velocity all create {:.2f} {:d}\n".format(T, np.random.randint(int(1E3), int(1E6)))
    string += "fix 1 all langevin {0:.2f} {0:.2f} {1:.1f} {2:d}\n".format(T, damping_const, np.random.randint(int(1E3), int(1E6)))
    string += "fix 2 all nve\n\n"
    return string, 2

def fix_Nose_Hoover_integrator(T, damping_const, tchain):
    string = "# Set integrator, temperature, damping constant.\n"
    string += "velocity all create {:.2f} {:d}\n".format(T, np.random.randint(int(1E5), int(1E6)))
    string += "fix 1 all nvt temp {0:.2f} {0:.2f} {1:.1f} tchain {2:d}\n\n".format(T, damping_const, tchain)
    return string, 1

def get_awsem_in_script(T, nsteps, topfile, seqfile, CA_idxs, CB_HB_idxs, O_idxs,
            boundary=["p","p","p"], n_steps_xtc=1000, integrator="langevin",
            damping_const=100., tchain=5, 
            trajname="traj.xtc", extra_group_defs="", extra_fix_defs=""):

    aw_string = header_string(boundary)
    aw_string += data_and_pair_coeff_string(topfile)
    aw_string += group_def_string(CA_idxs, CB_HB_idxs, O_idxs, extra_group_defs)

    if integrator == "Langevin":
        int_string, fixid = fix_langevin_integrator(T, damping_const)
    elif integrator == "Nose-Hoover":
        int_string, fixid = fix_Nose_Hoover_integrator(T, damping_const, tchain)

    fixid += 1
    aw_string += int_string
    aw_string += "# This fix sets the AWSEM force field\n"
    aw_string += "fix {} alpha_carbons backbone beta_atoms oxygens fix_backbone_coeff.data {}\n\n".format(fixid, seqfile)

    aw_string += "# Extra fixes for e.g. umbrella sampling.\n"
    aw_string += extra_fix_defs

    aw_string += "# Output\n"
    aw_string += "thermo {:d}\n".format(n_steps_xtc)
    aw_string += "dump 1 all xtc {:d} {}\n\n".format(n_steps_xtc, trajname)

    aw_string += "# Run simulations\n"
    aw_string += "reset_timestep  0\n"
    aw_string += "run {:d}\n".format(nsteps)

    return aw_string


if __name__ == "__main__":
    pass
