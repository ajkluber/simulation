import os
import argparse
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

def get_run_cmd():
    runcmd = \
"""grompp_sbm -n index.ndx -f run.mdp -c conf.gro -p topol.top -o topol_4.5.tpr 
if [ -e topol_4.5.tpr ]; then  
    mdrun_sbm -s topol_4.5.tpr -maxh 23.5
    if [ -e traj.xtc ]; then 
        python calcQtanh_umb.py
        g_energy_sbm -f ener.edr -o energyterms -xvg none << EOF
Bond
Angle
Proper-Dih.
Gaussian
Potential
EOF

    else
        echo "Something went wrong with simulation!"
    fi
else
    echo "Something went wrong with preparation!"
fi
"""
    return runcmd


def get_rst_cmd():
    rstcmd = \
"""  
mdrun_sbm -s topol_4.5.tpr -cpi state.cpt -maxh 23.5
if [ -e traj.xtc ]; then 
    python calcQtanh_umb.py
    g_energy_sbm -f ener.edr -o energyterms -xvg none << EOF
Bond
Angle
Proper-Dih.
Gaussian
Potential
EOF

else
    echo "Something went wrong with simulation!"
fi
"""
    return rstcmd

def get_python_contacts_script(switch_width):
    """Returns string that is python script to calculate Qtanh umbrella"""
    Qtanh_umb =\
"""import numpy as np

import simulation.calc.util as util

trajfile = "traj.xtc"
tanh_scale = %f
chunksize = 1000
topology = "Native.pdb"
periodic = False

pairs = np.loadtxt("native_contacts.ndx",skiprows=1,dtype=int) - 1
r0 = np.loadtxt("pairwise_params",usecols=(4,),skiprows=1)[1:2*pairs.shape[0]:2] + 0.1
widths = tanh_scale*np.ones(pairs.shape[0],float)
contact_params = (r0,widths)

# Parameterize contact-based reaction coordinate
contact_function = util.get_sum_contact_function(pairs,"tanh",contact_params,periodic=False)

# Calculate tanh contacts
qtanh = util.calc_coordinate_for_traj(trajfile,contact_function,topology,chunksize)

np.savetxt("Qtanh_umb.dat",qtanh)
""" % (switch_width)

    return Qtanh_umb 

def get_args():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--name',
            type=str,
            required=True,
            help='Name of ini file.')

    parser.add_argument('--pairs',
            type=str,
            required=True,
            help='Name of contacts file.')

    parser.add_argument('--switch_width',
            type=float,
            default=0.2,
            help='Switching width (nm) of contact bias. Opt.')

    parser.add_argument('--k_umb',
            type=float,
            default=0.01,
            help='Umbrella potential spring constant. Opt.')

    parser.add_argument('--umb_steps',
            type=int,
            default=3e4,
            help='Total number of simulation steps. Opt.')

    parser.add_argument('--n_steps',
            type=int,
            default=1e8,
            help='Total number of simulation steps. Opt.')

    parser.add_argument('--temperature',
            type=float,
            default=110,
            help='Temperature. Opt.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    name = args.name
    pairsfile = args.pairs
    T = args.temperature
    switch_width = args.switch_width
    k_umb = args.k_umb
    umb_steps = args.umb_steps
    nsteps = args.n_steps

    pairs = np.loadtxt(pairsfile,dtype=int)

    kappa = 2./switch_width

    # determine the centers of umbrellas
    umb_centers = range(10,pairs.shape[0] + 10,10)
    n_umbs = len(umb_centers)

    dirs = [ "umb_%d" % umb_centers[i] for i in range(n_umbs) ]
    jobnames = [ "umb_%d" % umb_centers[i] for i in range(n_umbs) ]
    runcmd = get_run_cmd()
    rstcmd = get_rst_cmd()
    calcQtanh_string = get_python_contacts_script(switch_width)

    mdpfiles = [ pjt.simulation.mdp.constant_temperature(str(T),str(nsteps)) for i in range(n_umbs) ]
    runslurms = [ simulation.slurm.make_script_string(runcmd,jobnames[i],partition="serial",mem_per_cpu="2G",walltime="03:00:00") for i in range(n_umbs) ]
    rstslurms = [ simulation.slurm.make_script_string(rstcmd,jobnames[i],partition="serial",mem_per_cpu="2G",walltime="03:00:00") for i in range(n_umbs) ]

    model,fitopts = mdb.inputs.load_model(name)
    r0 = mdb.models.pdb_parser.get_pairwise_distances(model.cleanpdb,pairs)
    umb_r0 = (r0/1.2) + 0.1 # smb_gmx automatically scales distances like LJ1210 contacts. So we compensate here.
    morefiles = [] 
    for i in range(n_umbs):
        umbrella_string = get_umbrella_params_string(pairs,umb_r0,umb_centers[i],k_umb,kappa,umb_steps)
        morefiles.append({"umbrella_params":umbrella_string,"calcQtanh_umb.py":calcQtanh_string})

    if not os.path.exists("test_%.2f" % T):
        os.mkdir("test_%.2f" % T)
    os.chdir("test_%.2f" % T)

    simulation.run.submit_simulations(model.save_simulation_files,dirs,mdpfiles,runslurms,rstslurms,morefiles=morefiles)

    with open("umbrella_last","w") as fout:
        fout.write("\n".join(dirs))

    os.chdir("..")
