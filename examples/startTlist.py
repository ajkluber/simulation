import os
import shutil
import numpy as np
import subprocess as sb

import model_builder as mdb
import project_tools as pjt

def get_slurm(jobname,queue="serial",email=False,analysis=False):
    slurm_string = "#!/bin/bash \n"
    slurm_string +="#SBATCH --job-name=%s\n" % jobname
    slurm_string +="#SBATCH --partition=serial\n"
    slurm_string +="#SBATCH --nodes=1\n"
    slurm_string +="#SBATCH --ntasks-per-node=1\n"
    slurm_string +="#SBATCH --time=23:55:00\n"
    slurm_string +="#SBATCH --mem=4G\n"
    #slurm_string +="#SBATCH --no-kill\n"
    if email:
        slurm_string +="#SBATCH --mail-user=alexkluber@gmail.com\n"
        slurm_string +="#SBATCH --mail-type=ae\n" 
    slurm_string +="#SBATCH --export=ALL\n\n"
    #slurm_string +="echo 'I ran on:'\n"
    slurm_string +="cd $SLURM_SUBMIT_DIR\n"
    slurm_string +="mdrun_sbm -s topol_4.5.tpr -maxh 23\n"
    if analysis:
        slurm_string +="""g_kuh_sbm -s Native.pdb -f traj.xtc -n native_contacts.ndx -o Q -noshortcut -abscut -cut 0.1
mv Q.out Q.dat
g_energy_sbm -f ener.edr -o energyterms -xvg none << EOF
Bond
Angle
Proper-Dih.
Gaussian
Potential
EOF"""
    return slurm_string

def get_rst_slurm(jobname,analysis=False):
    slurm_string = "#!/bin/bash \n"
    slurm_string +="#SBATCH --job-name=%s\n" % jobname
    slurm_string +="#SBATCH --partition=serial\n"
    slurm_string +="#SBATCH --nodes=1\n"
    slurm_string +="#SBATCH --ntasks-per-node=1\n"
    slurm_string +="#SBATCH --time=23:55:00\n"
    slurm_string +="#SBATCH --export=ALL\n\n"
    slurm_string +="#SBATCH --mem=4G\n"
    #slurm_string +="#SBATCH --no-kill\n"
    #slurm_string +="echo 'I ran on:'\n"
    #slurm_string +="cat $SLURM_JOB_NODELIST\n"
    slurm_string +="cd $SLURM_SUBMIT_DIR\n"
    slurm_string +="mdrun_sbm -s topol_4.5.tpr -cpi state.cpt -maxh 23\n"
    if analysis:
        slurm_string +="""g_kuh_sbm -s Native.pdb -f traj.xtc -n native_contacts.ndx -o Q -noshortcut -abscut -cut 0.1
cp Q.out Q.dat
g_energy_sbm -f ener.edr -o energyterms -xvg none << EOF
Bond
Angle
Proper-Dih.
Gaussian
Potential
EOF"""
    return slurm_string

def get_extend_rst_slurm(jobname,extend_ps,analysis=False):
    slurm_string = "#!/bin/bash \n"
    slurm_string +="#SBATCH --job-name=%s\n" % jobname
    slurm_string +="#SBATCH --partition=serial\n"
    slurm_string +="#SBATCH --nodes=1\n"
    slurm_string +="#SBATCH --ntasks-per-node=1\n"
    #slurm_string +="#SBATCH --time=24:00:00\n"    
    slurm_string +="#SBATCH --time=23:55:00\n"
    slurm_string +="#SBATCH --export=ALL\n\n"
    slurm_string +="#SBATCH --mem=4G\n"
    #slurm_string +="echo 'I ran on:'\n"
    #slurm_string +="cat $SLURM_JOB_NODELIST\n"
    slurm_string +="cd $SLURM_SUBMIT_DIR\n"
    slurm_string +="tpbconv_sbm -s topol_4.5.tpr -o topol_4.5_ext.tpr -extend %d\n" % extend_ps
    slurm_string +="mv topol_4.5_ext.tpr topol_4.5.tpr\n"
    slurm_string +="mdrun_sbm -s topol_4.5.tpr -cpi state.cpt -maxh 23\n"
    if analysis:
        slurm_string +="""g_kuh_sbm -s Native.pdb -f traj.xtc -n native_contacts.ndx -o Q -noshortcut -abscut -cut 0.1
cp Q.out Q.dat
g_energy_sbm -f ener.edr -o energyterms -xvg none << EOF
Bond
Angle
Proper-Dih.
Gaussian
Potential
EOF"""

    return slurm_string

def start_temperatures_Tlist(model,Tlist):

    nsteps = 100000000 
    name = model.name

    Tstring = ""
    for m in range(len(Tlist)):
        T = "%d" % Tlist[m]
        Tdir = "%d_0" % Tlist[m]
        Tstring += Tdir + "\n"

        if os.path.exists(Tdir):
            print "skipping ", Tdir
        else:
            print "starting ", Tdir
            os.mkdir(Tdir)
            os.chdir(Tdir)

            nvt = pjt.simulation.mdp.constant_temperature(T,str(nsteps))

            model.save_simulation_files()

            with open("nvt.mdp","w") as fout:
                # Molecular dynamics parameters
                fout.write(nvt)

            with open("prep.out","w") as fout:
                # Prepare run
                prep_step = 'grompp_sbm -n index.ndx -f nvt.mdp -c conf.gro -p topol.top -o topol_4.5.tpr '
                sb.call(prep_step.split(),stdout=fout,stderr=fout)

            with open("run.slurm","w") as fout:
                fout.write(get_slurm(name+T,analysis=True))

            with open("rst.slurm","w") as fout:
                fout.write(get_rst_slurm(name+T))

            with open("sim.out","w") as fout:
                # Submit simulation and get jobID
                sbatch = "sbatch run.slurm"
                sb.call(sbatch.split(),stdout=fout,stderr=fout)

            os.chdir("..")

    with open("short_temps_last","w") as fout:
        fout.write(Tstring)
    with open("short_temps","a") as fout:
        fout.write(Tstring)

def run_wham(name,iteration,long=False):
    os.chdir("%s/iteration_%d" % (name,iteration))
    pjt.analysis.wham.run_wham_for_heat_capacity()
    os.chdir("../..")

def start_long_temperatures(model,Tfguess,n_replicas=3,jobchain=False):

    Tlist = [Tfguess - 1, Tfguess, Tfguess + 1]
    nsteps = 550000000 
    extend_ps = int(0.5*(2E8/1000))
    name = model.name

    Tstring = ""
    for m in range(len(Tlist)):
        for n in range(1,n_replicas + 1):
            T = "%.2f" % Tlist[m]
            Tdir = "%.2f_%d" % (Tlist[m], n)
            Tstring += Tdir + "\n"

            if os.path.exists(Tdir):
                print "skipping ", Tdir
            else:
                print "starting ", Tdir
                os.mkdir(Tdir)
                os.chdir(Tdir)

                nvt = pjt.simulation.mdp.constant_temperature(T,str(nsteps))

                model.save_simulation_files()

                with open("nvt.mdp","w") as fout:
                    # Molecular dynamics parameters
                    fout.write(nvt)

                with open("prep.out","w") as fout:
                    # Prepare run
                    prep_step = 'grompp_sbm -n index.ndx -f nvt.mdp -c conf.gro -p topol.top -o topol_4.5.tpr '
                    sb.call(prep_step.split(),stdout=fout,stderr=fout)

                with open("run.slurm","w") as fout:
                    fout.write(get_slurm(name+T))

                with open("rst0.slurm","w") as fout:
                    fout.write(get_rst_slurm(name+T))

                with open("rst1.slurm","w") as fout:
                    fout.write(get_extend_rst_slurm(name+T,extend_ps))

                with open("rst2.slurm","w") as fout:
                    fout.write(get_extend_rst_slurm(name+T,extend_ps,analysis=True))

                # Submit simulation and two extension jobs that are held until 
                # the previous job finishes. Each leg is ~22hrs.
                with open("sim.out","w+") as fout:
                    # Submit simulation and get jobID
                    sbatch = "sbatch run.slurm"
                    sb.call(sbatch.split(),stdout=fout)
                    fout.seek(0,0)
                    job1 = int(fout.read().split()[-1])

                if jobchain:
                    with open("rst1.out","w+") as fout:
                        # Create dependent job that extends first leg.
                        sbatch = "sbatch --dependency=afterok:%d rst1.slurm" % job1
                        sb.call(sbatch.split(),stdout=fout)
                        fout.seek(0,0)
                        job2 = int(fout.read().split()[-1])

                    with open("rst2.out","w+") as fout:
                        # Create dependent job that extends second leg.
                        sbatch = "sbatch --dependency=afterok:%d rst2.slurm" % job2
                        sb.call(sbatch.split(),stdout=fout)

                os.chdir("..")

    with open("long_temps_last","w") as fout:
        fout.write(Tstring)
    with open("long_temps","a") as fout:
        fout.write(Tstring)

if __name__ == "__main__":
    Tlist = range(136,146,1) # Iteration 0
    #Tlist = range(140,141,1) # Iteration 0
    
    name = "PDZ"
    #native_variance = ['0.01','0.09','0.25','1.00']
    #native_variance = ['0.01','0.09']
    #native_variance = ['0.09']
    #native_variance = ['0.01']
    native_variance = ['0.25']
    #native_variance = ['0.01','0.09']
    #native_variance = ['0.25','1.00']
    #native_variance = ['1.00']
    #native_variance = ['0.09']

    #native_variance = ['0.49','0.64','0.81']
    #native_variance = ['0.49']
    #native_variance = ['0.64']
    #native_variance = ['0.81']

    replicas = range(1,11)
    #eplicas = range(4,11)
    #replicas = range(1,6)
    #replicas = range(7,11)

    # Larger window needed for large variance
    #Tlist = range(148,160,2) 
    #native_variance = ['1.00']
    #replicas = [1, 3, 4, 5, 7, 10]
    #replicas = [10]
    #replicas = [6]

    for n in range(len(native_variance)):
        os.chdir("random_native_%s" % native_variance[n])
        print "\nnative variance: ", native_variance[n]

        for i in replicas:
            os.chdir("replica_%d" % i)
            print "  replica: ", i

            model,fitopts = mdb.inputs.load_model(name)

            os.chdir("%s/iteration_%d" % (name,0))

            # Run short simulations
            #start_temperatures_Tlist(model,Tlist)

            # Run WHAM
            #pjt.analysis.wham.run_wham_for_heat_capacity()

            # Run long simulations
            shortTf = float(open("short_Tf","r").read().split("\n")[0])
            start_long_temperatures(model,shortTf)
            
            # Run long WHAM
            #pjt.analysis.wham.run_wham_for_heat_capacity(long=True)

            os.chdir("../..")
            os.chdir("..")
        os.chdir("..")

