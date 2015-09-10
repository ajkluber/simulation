import os
import subprocess as sb

def start_set_of_simulations(model,dirs,mdpfiles,runslurms,rstslurms,prepcmd=False):
    """Run simulations"""
    n_dirs = len(dirs)
    name = model.name

    if len(mdpfiles) == 1:
        mdpfiles = [ mdpfiles[0] for i in range(n_dirs) ]

    if prepcmd:
        prep_step = prepcmd
    else:
        prep_step = 'grompp_sbm -n index.ndx -f run.mdp -c conf.gro -p topol.top -o topol_4.5.tpr'

    for n in range(n_dirs):
        dir = dirs[n]

        if os.path.exists(dir):
            print "skipping ", dir
        else:
            print "starting ", dir
            os.mkdir(Tdir)
            os.chdir(Tdir)
            mdp = mdpfiles[n]
            runslurm = runslurms[n]
            rstslurm = rstslurms[n]

            model.save_simulation_files()

            with open("run.mdp","w") as fout:
                # Molecular dynamics parameters
                fout.write(mdp)

            with open("prep.out","w") as fout:
                # Prepare run
                sb.call(prep_step.split(),stdout=fout,stderr=fout)

            with open("run.slurm","w") as fout:
                fout.write(runslurm)

            with open("rst.slurm","w") as fout:
                fout.write(rstslurm)

            with open("sim.out","w") as fout:
                # Submit simulation and get jobID
                sbatch = "sbatch run.slurm"
                sb.call(sbatch.split(),stdout=fout,stderr=fout)

            os.chdir("..")

