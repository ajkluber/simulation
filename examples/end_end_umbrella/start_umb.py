import os
import numpy as np
import subprocess as sb

import model_builder as mdb

import simulation.mdp
import simulation.slurm

def save_ref_traj(model):
    # Save the starting configuration, but we have to fix the unitcell
    # information.
    model.ref_traj[0].save("conf.gro")
    with open("conf.gro", "r") as fin:
        temp = reduce(lambda x,y: x+y, fin.readlines()[:-1])
        temp += "{:>10f}{:>10f}{:>10f}\n".format(10,10,10)

    with open("conf.gro", "w") as fout:
        fout.write(temp)

    # Save a pdb with conect directives for ease of visualization.
    model.ref_traj[0].save("ca.pdb")
    mdb.models.structure.viz_bonds.write_bonds_conect(model.mapping.top)
    with open("ca.pdb", "r") as fin:
        temp = reduce(lambda x,y: x+y, fin.readlines()[:-3])
        temp += open("conect.pdb","r").read() + "END\n"
    with open("ca.pdb", "w") as fout:
        fout.write(temp)


if __name__ == "__main__":
    KB_KJ_MOL = 0.0083145

    run_commands = \
"""grompp_sbm -f run.mdp -n index.ndx -c conf.gro -p topol.top -o topol.tpr
mdrun_sbm -s topol.tpr -table table.xvg -tablep tablep.xvg
g_energy_sbm -f ener.edr -o Etot -xvg none << HERE
Potential
HERE
mv Etot.xvg Etot.dat
"""

    # Umbrella sampling along end-to-end distance
    r1N_centers = np.linspace(1, 10, 10)
    T = 129.0
    kumb = 2

    # create model from saved settings
    model, fitopts = mdb.inputs.load_model("SH3.ini")

    # Create the Hamiltonian Gromacs input file: topol.top
    writer = mdb.models.output.GromacsFiles(model)
    writer.generate_topology()

    # Add groups to pull between
    ndx = writer.index_ndx
    ndx += "[ Handle1 ]\n"
    ndx += "{:>4d}\n".format(1)
    ndx += "[ Handle2 ]\n"
    ndx += "{:>4d}\n".format(model.mapping.top.n_atoms)

    for i in range(len(r1N_centers)):
    #for i in [3]:
        # start umbrella for separation
        if not os.path.exists("{:.2f}".format(r1N_centers[i])):
            os.mkdir("{:.2f}".format(r1N_centers[i]))
        os.chdir("{:.2f}".format(r1N_centers[i]))

        # Save starting configuration
        save_ref_traj(model)

        # Save topology file
        with open("topol.top", "w") as fout:
            fout.write(writer.topfile)

        np.savetxt("table.xvg", writer.tablep, fmt="%16.15e")
        np.savetxt("tablep.xvg", writer.tablep, fmt="%16.15e")

        with open("index.ndx", "w") as fout:
            fout.write(ndx)

        # running a constant temperature simulation
        nvt_mdp = simulation.mdp.constant_temperature(T, "100000000")

        # Add pull code section to mdp file
        pull_section = simulation.mdp.pull_code("Handle1", "Handle2", kumb, r1N_centers[i])
        nvt_mdp += "{}\n".format(pull_section) 

        with open("run.mdp", "w") as fout:
            fout.write(nvt_mdp)

        # Run simulation
        with open("run.slurm", "w") as fout:
            slurm_string = simulation.slurm.make_script_string(run_commands, "pull_{}".format(i), walltime="08:00:00")
            fout.write(slurm_string)

        sb.call("sbatch run.slurm", shell=True)

        os.chdir("..")
