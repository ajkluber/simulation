import os
import numpy as np
import subprocess as sb

import model_builder as mdb
import project_tools as pjt

def get_pbs(jobname,queue="serial",email=False,analysis=False):
    pbs_string = "#!/bin/bash \n"
    pbs_string +="#PBS -N %s\n" % jobname
    pbs_string +="#PBS -q %s\n" % queue
    pbs_string +="#PBS -l nodes=1:ppn=1\n"
    pbs_string +="#PBS -l walltime=10:00:00\n"
    if email:
        pbs_string +="#PBS -M alexkluber@gmail.com\n"
        pbs_string +="#PBS -m ae\n"
    pbs_string +="#PBS -V \n\n"
    pbs_string +="echo 'I ran on:'\n"
    pbs_string +="cat $PBS_NODEFILE\n"
    pbs_string +="cd $PBS_O_WORKDIR\n"
    pbs_string +="mdrun_sbm -s topol_4.5.tpr -maxh 23\n"
    if analysis:
        pbs_string +="""g_kuh_sbm -s Native.pdb -f traj.xtc -n native_contacts.ndx -o Q -noshortcut -abscut -cut 0.1
mv Q.out Q.dat
g_kuh_sbm -s Native.pdb -f traj.xtc -n native_contacts.ndx -o Qtanh -kappa 5
mv Qtanh.out Qtanh.dat
g_energy_sbm -f ener.edr -o energyterms -xvg none << EOF
Bond
Angle
Proper-Dih.
Gaussian
Potential
EOF"""
    return pbs_string

def get_rst_pbs(jobname,analysis=False):
    pbs_string = "#!/bin/bash \n"
    pbs_string +="#PBS -N %s \n" % jobname
    pbs_string +="#PBS -q serial \n"
    pbs_string +="#PBS -l nodes=1:ppn=1 \n"
    pbs_string +="#PBS -l walltime=24:00:00 \n"
    pbs_string +="#PBS -V \n\n"
    pbs_string +="echo 'I ran on:'\n"
    pbs_string +="cat $PBS_NODEFILE\n"
    pbs_string +="cd $PBS_O_WORKDIR\n"
    pbs_string +="mdrun_sbm -s topol_4.5.tpr -cpi state.cpt -maxh 23\n"
    if analysis:
        pbs_string +="""g_kuh_sbm -s Native.pdb -f traj.xtc -n native_contacts.ndx -o Q -noshortcut -abscut -cut 0.1
mv Q.out Q.dat
g_kuh_sbm -s Native.pdb -f traj.xtc -n native_contacts.ndx -o Qtanh -kappa 5
mv Qtanh.out Qtanh.dat
g_energy_sbm -f ener.edr -o energyterms -xvg none << EOF
Bond
Angle
Proper-Dih.
Gaussian
Potential
EOF"""
    return pbs_string

if __name__ == "__main__":

    name = "SH3"
    contacts = np.loadtxt("SH3.contacts",dtype=int)

    kappa = 5
    k_umb = 0.1
    Qcenter = range(20,120,20)
    n_conts = len(contacts)
    umb_steps = 20000

    T = 130.95
    nsteps = 100000000

    model,fitopts = mdb.inputs.load_model(name)

    distances = mdb.models.pdb_parser.get_pairwise_distances(name+".pdb",contacts)
    body_string = ""
    for i in range(len(contacts)):
        body_string += "%5d %5d %.10f\n" % (contacts[i,0],contacts[i,1],distances[i])

    if not os.path.exists("kumb_%.2f" % k_umb):
        os.mkdir("kumb_%.2f" % k_umb)
    os.chdir("kumb_%.2f" % k_umb)

    print "starting kumb_%.2f" % k_umb

    umb_last = ""
    for i in range(len(Qcenter)):
        Q0 = Qcenter[i]
        if not os.path.exists(str(Q0)):
            os.mkdir(str(Q0))
        os.chdir(str(Q0))
        print "running ", str(Q0)

        umb_last += str(Q0) + "\n"

        umb_string = "%.4f %.4f %.4f %d %d\n" % (Q0,k_umb,kappa,n_conts,umb_steps)
        umb_string += body_string
        with open("umbrella_params","w") as fout:
            fout.write(umb_string)

        nvt = pjt.simulation.mdp.constant_temperature(str(T),str(nsteps))

        model.save_simulation_files()

        with open("nvt.mdp","w") as fout:
            # Molecular dynamics parameters
            fout.write(nvt)

        with open("prep.out","w") as fout:
            # Prepare run
            prep_step = 'grompp_sbm -n index.ndx -f nvt.mdp -c conf.gro -p topol.top -o topol_4.5.tpr '
            sb.call(prep_step.split(),stdout=fout,stderr=fout)

        with open("run.pbs","w") as fout:
            fout.write(get_pbs(name+"umb"+str(Q0),analysis=True))

        with open("rst.pbs","w") as fout:
            fout.write(get_rst_pbs(name+"umb"+str(Q0),analysis=True))

        with open("sim.out","w") as fout:
            # Submit simulation 
            qsub = "qsub run.pbs"
            sb.call(qsub.split(),stdout=fout,stderr=fout)

        os.chdir("..")

    with open("umbrella_last","w") as fout:
        fout.write(umb_last)

    os.chdir("..")
