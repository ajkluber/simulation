import os
import numpy as np
import subprocess as sb

import model_builder as mdb
import project_tools as pjt

import simulation.examples.constant_temp as constT
import simulation.slurm
import simulation.mdp
import simulation.run

def get_umbrella_params_string(pairs,r0,cent_umb,k_umb,kappa,umb_steps):
    umb_string = "%.4f %.4f %.4f %d %d\n" % (cent_umb,k_umb,kappa,pairs.shape[0],umb_steps)
    for i in range(pairs.shape[0]):
        umb_string += "%5d %5d %.10f\n" % (pairs[i,0],pairs[i,1],r0[i])
    return umb_string

def get_args():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--name',
            type=str,
            required=True,
            help='Name of ini file.')

    parser.add_argument('--pairsfile',
            type=str,
            required=True,
            help='Name of contacts file.')

    parser.add_argument('--kappa',
            type=str,
            default="Native.pdb",
            help='Contact functional form. Opt.')

    args = parser.parse_args()
    return args

def get_run_cmd(kappa,postruncmd=""):
    runcmd = \
"""grompp_sbm -n index.ndx -f run.mdp -c conf.gro -p topol.top -o topol_4.5.tpr 
if [ -e topol_4.5.tpr ]; then  
    mdrun_sbm -s topol_4.5.tpr -maxh 23.5
    if [ -e traj.xtc ]; then 
        g_kuh_sbm -s Native.pdb -f traj.xtc -n native_contacts.ndx -o Qumb -kappa %f
        mv Qumb.out Qumb.dat
        g_energy_sbm -f ener.edr -o energyterms -xvg none << EOF
Bond
Angle
Proper-Dih.
Gaussian
Potential
EOF
    %s
    else
        echo "Something went wrong with simulation!"
    fi
else
    echo "Something went wrong with preparation!"
fi
""" % (kappa,postruncmd)
    return runcmd


def get_rst_cmd(kappa,postruncmd="",noappend=False):
    if noappend:
        noappend = "-noappend"
    else:
        noappend = ""
    rstcmd = \
"""  
mdrun_sbm -s topol_4.5.tpr -cpi state.cpt -maxh 23.5 %s
if [ -e traj.xtc ]; then 
    g_kuh_sbm -s Native.pdb -f traj.xtc -n native_contacts.ndx -o Qumb -kappa %f
    mv Qumb.out Qumb.dat
    g_energy_sbm -f ener.edr -o energyterms -xvg none << EOF
Bond
Angle
Proper-Dih.
Gaussian
Potential
EOF
    %s
else
    echo "Something went wrong with simulation!"
fi
""" % (noappend,kappa,postruncmd)
    return rstcmd


if __name__ == "__main__":

    #args = get_args()
    #name = args.name
    #pairs = np.loadtxt(args.pairsfile,dtype=int)

    name = "PDZ"
    pairs = np.loadtxt("cacb_cutoff_contacts",dtype=int)

    # Needed:
    kappa = 5
    k_umb = 0.1
    umb_steps = 2e4
    
    T = 120
    nsteps = 1e8

    # determine the centers of umbrellas
    umb_centers = range(20,pairs.shape[0],20)
    n_umbs = len(umb_centers)

    model,fitopts = mdb.inputs.load_model(name)

    r0 = mdb.models.pdb_parser.get_pairwise_distances(model.cleanpdb,pairs)

    dirs = [ "umb_%d" % umb_centers[i] for i in range(n_umbs) ]
    jobnames = [ "umb_%d" % umb_centers[i] for i in range(n_umbs) ]
    umbrellafiles = [ {"umbrella_params":get_umbrella_params_string(pairs,r0,umb_centers[i],k_umb,kappa,umb_steps)} for i in range(n_umbs) ]
    runcmd = get_run_cmd(kappa)
    rstcmd = get_rst_cmd(kappa)

    mdpfiles = [ pjt.simulation.mdp.constant_temperature(str(T),str(nsteps)) for i in range(n_umbs) ]
    runslurms = [ simulation.slurm.make_script_string(runcmd,jobnames[i],partition="serial",mem_per_cpu="2G",walltime="00:10:00") for i in range(n_umbs) ]
    rstslurms = [ simulation.slurm.make_script_string(rstcmd,jobnames[i],partition="serial",mem_per_cpu="2G") for i in range(n_umbs) ]

    if not os.path.exists("test_%.2f" % T):
        os.mkdir("test_%.2f" % T)
    os.chdir("test_%.2f" % T)

    simulation.run.submit_simulations(model.save_simulation_files,dirs,mdpfiles,runslurms,rstslurms,morefiles=umbrellafiles)

    with open("umbrella_last","w") as fout:
        fout.write("\n".join(dirs))

    os.chdir("..")
