import os
import pdb
import shutil
import time
import numpy as np
import subprocess as sb

import simulation.slurm
import simulation.run
import simulation.mdp
import model_builder as mdb
import project_tools as pjt

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def extend_long_temperatures(model,jobpreffix):

    nsteps = 100000000 
    extend_ps = 2E5
    name = model.name
    want_ps = 1E6

    Tlist = [ x.rstrip("\n") for x in open("long_temps","r").readlines() ]
    for m in range(len(Tlist)):
        Tdir = Tlist[m]

        os.chdir(Tdir)

        completed_ps = 0.5*float(file_len("Q.dat"))
        extend_ps = want_ps - completed_ps
        #assert extend_ps >= 0, " extending number of steps should be non-negative"
        if (extend_ps <= 0):
            os.chdir("..")
            continue
        else:
            print "extending %s  from %e to %e ps" % (Tdir,completed_ps,want_ps)
            #model.save_simulation_files()
            jobname = jobpreffix + Tdir
            with open("extend.slurm","w") as fout:
                fout.write(get_extend_rst_slurm(jobname,extend_ps,noappend=True))

            with open("rst.slurm","w") as fout:
                fout.write(get_rst_slurm(jobname,noappend=True))

            with open("sim.out","w") as fout:
                # Submit simulation and get jobID
                sbatch = "sbatch extend.slurm"
                sb.call(sbatch.split(),stdout=fout,stderr=fout)
            os.chdir("..")

def start_long_temperatures(model,Tfguess,n_replicas=3,jobchain=False):

    Tlist = [Tfguess - 1, Tfguess, Tfguess + 1]
    nsteps = 550000000 
    nsteps = 1000000000 
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
                    fout.write(get_slurm(name+T,partition="commons"))

                with open("rst0.slurm","w") as fout:
                    fout.write(get_rst_slurm(name+T,partition="commons"))
#                # Submit simulation and two extension jobs that are held until 
#                # the previous job finishes. Each leg is ~22hrs.
                with open("sim.out","w+") as fout:
                    # Submit simulation and get jobID
                    sbatch = "sbatch run.slurm"
                    sb.call(sbatch.split(),stdout=fout)
                    fout.seek(0,0)
                    job1 = int(fout.read().split()[-1])

                os.chdir("..")

    with open("long_temps_last","w") as fout:
        fout.write(Tstring)
    with open("long_temps","a") as fout:
        fout.write(Tstring)

def get_run_cmd(postruncmd=""):
    runcmd = \
"""grompp_sbm -n index.ndx -f run.mdp -c conf.gro -p topol.top -o topol_4.5.tpr 
if [ -e topol_4.5.tpr ]; then  
    mdrun_sbm -s topol_4.5.tpr -maxh 23.5
    if [ -e traj.xtc ]; then 
        g_kuh_sbm -s Native.pdb -f traj.xtc -n native_contacts.ndx -o Q -noshortcut -abscut -cut 0.1
        mv Q.out Q.dat
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
""" % postruncmd
    return runcmd

def get_rst_cmd(postruncmd="",noappend=False):
    if noappend:
        noappend = "-noappend"
    else:
        noappend = ""
    rstcmd = \
"""  
mdrun_sbm -s topol_4.5.tpr -cpi state.cpt -maxh 23.5 %s
if [ -e traj.xtc ]; then 
    g_kuh_sbm -s Native.pdb -f traj.xtc -n native_contacts.ndx -o Q -noshortcut -abscut -cut 0.1
    mv Q.out Q.dat
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
""" % (postruncmd,noappend)
    return rstcmd

def get_extend_cmd(extend_ps,noappend=False):
    if noappend:
        noappend = "-noappend"
    else:
        noappend = ""
    extendcmd = \
"""  
tpbconv_sbm -s topol_4.5.tpr -o topol_4.5_ext.tpr -extend %.2f
mv topol_4.5_ext.tpr topol_4.5.tpr
mdrun_sbm -s topol_4.5.tpr -cpi state.cpt -maxh 23.5 %s
if [ -e traj.xtc ]; then 
    g_kuh_sbm -s Native.pdb -f traj.xtc -n native_contacts.ndx -o Q -noshortcut -abscut -cut 0.1
    mv Q.out Q.dat
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
""" % (extend_ps,noappend)
    return extendcmd

def get_check_cmd(Qfolded,extend_ps,noappend=False):
    if noappend:
        noappend = "-noappend"
    else:
        noappend = ""
    extendcmd = \
"""# Bash script to see if protein folded and if not extend the simulation
folded=0
while read p; do
    if [ $p -ge %d ]
    then
        echo 'Protein has folded! Stopping simulation'
        folded=1
        break
    fi
done <Q.dat
if [ ${folded} -eq 0 ]
then
    tpbconv_sbm -s topol_4.5.tpr -o topol_4.5_ext.tpr -extend %f
    mv topol_4.5_ext.tpr topol_4.5.tpr 
    mdrun_sbm -s topol_4.5.tpr -cpi state.cpt -maxh 23.5 %s
    g_kuh_sbm -s Native.pdb -f traj.xtc -n native_contacts.ndx -o Q -noshortcut -abscut -cut 0.1
    mv Q.out Q.dat
    g_energy_sbm -f ener.edr -o energyterms -xvg none << EOF
Bond
Angle
Proper-Dih.
Gaussian
Potential
EOF
fi""" % (Qfolded,extend_ps,noappend)
    return extendcmd

    #jobinfo=`sbatch rst.slurm`
    ## Chain a dependent job that repeats the checking.
    #IFS=' ' read -a jobinfo2 <<< "$jobinfo"
    #jobID=`echo "${jobinfo2[3]}"`
    #sbatch -W depend=afterany:${jobID} check.slurm

if __name__ == "__main__":
    #Tlist = range(136,146,1) # Iteration 0
    #Tlist = range(140,141,1) # Iteration 0
    Tlist = range(136,138,1) # Iteration 0
    n_temps = len(Tlist)

    runcmd = get_run_cmd()
    rstcmd = get_rst_cmd()

    #extend_ps = 100000
    #extendcmd = get_extend_cmd(extend_ps)

    nsteps = 100000000 

    name = "PDZ"

    model,fitopts = mdb.inputs.load_model(name)

    os.chdir("%s/iteration_%d" % (name,0))


    dirs = [ "%d_0" % Tlist[j] for j in range(n_temps) ]
    jobnames = [ "%s_%d_%d" % (native_variance[n],i,Tlist[j]) for j in range(n_temps) ]
    runslurms = [ simulation.slurm.make_script_string(runcmd,jobnames[j],mem_per_cpu="3G") for j in range(n_temps) ]
    rstslurms = [ simulation.slurm.make_script_string(rstcmd,jobnames[j],mem_per_cpu="3G") for j in range(n_temps) ]
    mdpfiles = [ simulation.mdp.constant_temperature(str(Tlist[j]),str(nsteps)) for j in range(n_temps) ]

    simulation.run.submit_simulations(model.save_simulation_files,dirs,mdpfiles,runslurms,rstslurms)

    short_temps_last = ""
    for j in range(n_temps):
        short_temps_last += "%d\n" % Tlist[j]
    with open("short_temps_last","w") as fout:
        fout.write(short_temps_last)
    with open("short_temps","wa") as fout:
        fout.write(short_temps_last)

    # Extend long_temps simulations
    #extend_long_temperatures(model,jobpreffix)

    # Run WHAM
    #pjt.analysis.wham.run_wham_for_heat_capacity()

    # Run long simulations
    #shortTf = float(open("short_Tf","r").read().split("\n")[0])
    #start_long_temperatures(model,shortTf)
    
    # Run long WHAM
    #pjt.analysis.wham.run_wham_for_heat_capacity(long=True)

    os.chdir("../..")

