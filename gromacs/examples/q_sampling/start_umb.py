import os
import numpy as np
import subprocess as sb

import mdtraj as md


import model_builder as mdb
import simulation.mdp
import simulation.slurm

if __name__ == "__main__":
    model, fitopts = mdb.inputs.load_model("SH3.ini")

    T = 130.95
    nsteps = int(1E8)
    #nsteps = int(1E6)

    # Parameters for Umbrella sampling along Q.
    gamma = 5
    k_umb = 0.01
    Qcenter = np.linspace(10, 120, 10)
    umb_steps = 10000

    pairs = np.loadtxt("SH3.contacts", dtype=int)
    n_pairs = len(pairs)
    pair_r0 = md.compute_distances(model.ref_traj, pairs - 1)[0]

    body_string = ""
    for i in range(n_pairs):
        body_string += "%5d %5d %.10f\n" % (pairs[i,0], pairs[i,1], pair_r0[i])

    if not os.path.exists("kumb_%.2f" % k_umb):
        os.mkdir("kumb_%.2f" % k_umb)
    os.chdir("kumb_%.2f" % k_umb)

    print "starting kumb_%.2f" % k_umb

    # Gromacs topology writer
    writer = mdb.models.output.GromacsFiles(model)

    # Simulation commands
    nvt_mdp = simulation.mdp.constant_temperature(T, nsteps)

    run_commands = """grompp_sbm -f run.mdp -c conf.gro -p topol.top -o topol.tpr
mdrun_sbm -s topol.tpr -table table.xvg -tablep tablep.xvg
g_energy_sbm -f ener.edr -o Etot -xvg none <<HERE
Potential
HERE
mv Etot.xvg Etot.dat
"""

    umb_last = ""
    for i in range(len(Qcenter)):
        if not os.path.exists("{:.2f}".format(Qcenter[i])):
            os.mkdir("{:.2f}".format(Qcenter[i]))
        os.chdir("{:.2f}".format(Qcenter[i]))
        print "running {:.2f}".format(Qcenter[i])

        umb_last += "{:.2f}\n".format(Qcenter[i])

        with open("umbrella_params","w") as fout:
            umb_string = "{:.4f} {:.4f} {:.4f} {:d} {:d}\n".format(Qcenter[i], k_umb, gamma, n_pairs, umb_steps)
            umb_string += body_string
            fout.write(umb_string)

        with open("run.mdp","w") as fout:
            # Molecular dynamics parameters
            fout.write(nvt_mdp)

        writer.write_simulation_files()

        # Run simulation
        with open("run.slurm", "w") as fout:
            fout.write(simulation.slurm.make_script_string(run_commands, \
                                "qumb_{}".format(i), walltime="08:00:00"))

        sb.call("sbatch run.slurm", shell=True)

        os.chdir("..")

    with open("umbrella_last","w") as fout:
        fout.write(umb_last)

    os.chdir("..")
