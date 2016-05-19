import os 
import shutil
import subprocess as sb

import model_builder as mdb

import input_script

class AwsemParameters(object):
    """class to hold and set parameters for awsem"""
    def __init__(self):
        pass

        # Should have default values.

        # methods for setting commonly changed values.
        
        # method to construct the fix_backbone_coeff.dat file


#def get_fragment_memory_file(traj, path_to_fraglib):
#    """Write fragment memory file"""
#
#    frag_length = 9
#    frag_weight = 1
#
#    mem_string = "[Memories]\n"
#    for i in range(traj.n_chains):
#        chain = traj.top.chain(i)
#        n_res = chain.n_residues
#        # enumerate fragments in chain
#        for j in range(1, n_res + 1, frag_length):
#            frag_idx = chain.residue(j - 1).index + 1
#            if (frag_idx + frag_length) >= n_res:
#                frag_length = n_res - frag_idx
#            mem_string += "{} {:d} {:d} {:d} {:d}\n".format(
#                    path_to_fraglib, frag_idx, frag_idx, frag_length, frag_weight)
#
#    return mem_string

def get_fix_backbone_coeff(debye=True, frag_mem_file=None, frag_mem_strength=1):
    coeff_string = ""
    coeff_string += get_chain()
    coeff_string += get_chi()
    coeff_string += get_excluded()
    coeff_string += get_epsilon()
    coeff_string += get_rama()
    coeff_string += get_rama_p()
    coeff_string += get_ssweight()
    coeff_string += get_abc()
    coeff_string += get_dssp_hydrgn()
    coeff_string += get_p_ap()
    coeff_string += get_water()
    coeff_string += get_burial()
    coeff_string += get_helix()
    if not (frag_mem_file is None) and not (frag_mem_strength is None):
        coeff_string += get_fragment_memory(frag_mem_strength, frag_mem_file)
    if debye:
        coeff_string += get_debye_huckel()
    
    return coeff_string

def get_chain():
    return "[Chain]\n 10.0 10.0 30.0\n 2.45798 2.50665 2.44973\n\n"

def get_shake():
    return "[Shake]\n10.0 3.83 2.36 2.71\n\n"

def get_chi():
    return "[Chi]\n20.0 -0.83\n\n"

def get_excluded():
    return "[Excluded]\n5.0 3.0\n5.0 4.0\n\n"

def get_epsilon():
    return "[Epsilon]\n1.0\n\n"

def get_rama():
    return """[Rama]
2.0
5
 1.3149  15.398 0.15   1.74 0.65 -2.138
1.32016 49.0521 0.25  1.265 0.45  0.318
 1.0264 49.0954 0.65 -1.041 0.25  -0.78
    2.0   419.0  1.0  0.995  1.0  0.820
    2.0  15.398  1.0   2.25  1.0  -2.16\n\n"""

def get_rama_p():
    return """[Rama_P]
3
 0.0    0.0 1.0   0.0  1.0   0.0
2.17 105.52 1.0 1.153 0.15  -2.4
2.15 109.09 1.0  0.95 0.15 0.218
 0.0    0.0 1.0   0.0  2.0   0.0
 0.0    0.0 1.0   0.0  2.0   0.0\n\n"""

def get_ssweight():
    return "[SSWeight]\n0 0 0 1 1 0\n0 0 0 0 0 0\n\n"

def get_abc():
    return "[ABC]\n0.483 0.703 -0.186\n0.444 0.235 0.321\n0.841 0.893 -0.734\n\n"

def get_dssp_hydrgn():
    return """[Dssp_Hdrgn]
1.0
0.0  0.0
1.37  0.0  3.49 1.30 1.32 1.22   0.0
1.36  0.0  3.50 1.30 1.32 1.22   3.47  0.33 1.01
1.17  0.0  3.52 1.30 1.32 1.22   3.62  0.33 1.01
0.76   0.68
2.06   2.98
7.0
1.0    0.5
12.0\n\n"""

def get_p_ap():
    return """[P_AP]
1.0
1.5
1.0 0.4 0.4
8.0
7.0
5 8
4\n\n"""

def get_water():
    return """[Water]
1.0
5.0 7.0
2.6
13
2
4.5 6.5 1
6.5 9.5 1\n\n"""

def get_burial():
    return """[Burial]
1.0
4.0
0.0 3.0
3.0 6.0
6.0 9.0\n\n"""


def get_helix():
    return """[Helix]
1.5
2.0 -1.0
7.0 7.0
3.0
4
15.0
4.5 6.5
0.77 0.68 0.07 0.15 0.23 0.33 0.27 0.0 0.06 0.23 0.62 0.65 0.50 0.41 -3.0 0.35 0.11 0.45 0.17 0.14
0 -3.0
0.76   0.68
2.06   2.98\n\n"""

def get_fragment_memory(frag_mem_strength, memfile):
    return """[Fragment_Memory]
{:.8f}
{}
uniform.gamma\n\n""".format(frag_mem_strength, memfile)

def get_debye_huckel():
    return """[DebyeHuckel]
1.0 1.0 1.0
1.0
10.0
10\n\n"""

def get_solvent_barrier():
    return """[Solvent_Barrier]
1.0
4.5 6.5
1.0
6.0 7.0
5.0
13
1
0.00 2.04 0.57 0.57 0.36 1.11 1.17 -1.52 0.87 0.67 0.79 1.47 1.03 1.00 -0.10 0.26 0.37 1.21 1.15 0.39\n\n"""

def prep_constant_temp(model, traj, name, T, n_steps, n_steps_out, frag_strength, frag_mem_string,
                debye=False, awsem_other_param_files=["anti_HB", 
                "anti_NHB", "anti_one", "burial_gamma.dat",  "gamma.dat",
                "para_HB", "para_one", "uniform.gamma"], 
                awsem_param_path="/home/alex/packages/awsemmd/parameters", 
                extra_group_defs="", extra_fix_defs="", damping_const=10., tchain=5):

    seqfile = "{}.seq".format(name)
    memfile = "{}.mem".format(name)
    topfile = "data.{}".format(name)
    infile = "{}.in".format(name)

    CA_idxs = [ atom.index + 1 for atom in model.mapping.top.atoms if (atom.name == "CA") ] 
    CB_idxs = [ atom.index + 1 for atom in model.mapping.top.atoms if (atom.name in ["CB", "HB"]) ]
    O_idxs = [ atom.index + 1 for atom in model.mapping.top.atoms if (atom.name == "O") ] 

    # Save initial conditions
    model.starting_traj[0].save("ref.gro")
    model.starting_traj[0].save("ref.pdb")
    mdb.models.mappings.viz_bonds.write_bonds_tcl(model.mapping.top)

    # Write awsem parameter files
    writer = mdb.models.output.AWSEMLammpsFiles(model)
    writer.write_simulation_files(traj, topfile, seqfile)

    for filename in awsem_other_param_files:
        shutil.copy("{}/{}".format(awsem_param_path, filename), ".")

    with open(memfile, "w") as fout:
        fout.write(frag_mem_string)

    with open("fix_backbone_coeff.data", "w") as fout:
        fout.write(get_fix_backbone_coeff(debye=False, frag_mem_file=memfile, frag_mem_strength=frag_strength))

    # Simulation instructions file
    with open(infile, "w") as fout:
        lammps_in_string = input_script.get_awsem_in_script(T, n_steps,
                                topfile, seqfile, CA_idxs, CB_idxs, O_idxs,
                                n_steps_xtc=n_steps_out, 
                                extra_group_defs=extra_group_defs,
                                extra_fix_defs=extra_fix_defs,
                                integrator="Langevin",
                                damping_const=damping_const, tchain=tchain)

        fout.write(lammps_in_string)

