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


def get_awsem_in_script(T, nsteps, topfile, seqfile, CA_idxs, CB_HB_idxs, O_idxs,
            boundary=["p","p","p"], n_steps_xtc=1000, trajname="traj.xtc"):

    aw_string = header_string(boundary)

    aw_string += "# Read in the molecular topology\n"
    aw_string += "read_data {}\n\n".format(topfile)

    aw_string += "# Set the excluded volume of different pairs of atom types?\n"
    aw_string += "pair_coeff * * 0.0\n"
    aw_string += "pair_coeff 1 1 20.0 3.5 4.5\n"
    aw_string += "pair_coeff 1 4 20.0 3.5 4.5\n"
    aw_string += "pair_coeff 4 4 20.0 3.5 4.5\n"
    aw_string += "pair_coeff 3 3 20.0 3.5 3.5\n\n"

    aw_string += "# Define atom groups\n"
    aw_string += "group       alpha_carbons id {}\n".format(" ".join([ str(x) for x in CA_idxs ]))
    aw_string += "group       beta_atoms id {}\n".format(" ".join([ str(x) for x in CB_HB_idxs ]))
    aw_string += "group       oxygens id {}\n\n".format(" ".join([ str(x) for x in O_idxs ]))

    aw_string += "# Set integrator, temperature, damping constant.\n"
    aw_string += "velocity    all create {:.2f} 2349852\n".format(T)
    aw_string += "fix       1 all nvt temp {0:.2f} {0:.2f} 100.0\n".format(T)
    aw_string += "fix       2 alpha_carbons backbone beta_atoms oxygens fix_backbone_coeff.data {}\n\n".format(seqfile)

    aw_string += "# Output\n"
    aw_string += "thermo      {:d}\n".format(n_steps_xtc)
    aw_string += "dump        1 all xtc {:d} {}\n\n".format(n_steps_xtc, trajname)

    aw_string += "# Run simulations\n"
    aw_string += "reset_timestep  0\n"
    aw_string += "run     {:d}\n".format(nsteps)

    return aw_string


if __name__ == "__main__":
    pass
