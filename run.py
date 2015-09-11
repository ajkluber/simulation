import os
import subprocess as sb

def submit_simulations(save_simulation_files,dirs,mdpfiles,runslurms,rstslurms):
    """Run simulations 
    
    Parameters
    ----------
    model : object
        
    dirs : list
    
    mdpfiles : list

    runslurms : list

    rstslurms : list

    """
    n_dirs = len(dirs)
    for n in range(n_dirs):
        dir = dirs[n]
        if os.path.exists(dir):
            print "skipping %s" % dir
        else:
            print "starting %s" % dir
            os.mkdir(dir)
            os.chdir(dir)

            save_simulation_files()

            with open("run.mdp","w") as fout:
                fout.write(mdpfiles[n])

            with open("run.slurm","w") as fout:
                fout.write(runslurms[n])

            with open("rst.slurm","w") as fout:
                fout.write(rstslurms[n])

            with open("sim.out","w") as fout:
                sb.call("sbatch run.slurm".split(),stdout=fout,stderr=fout)

            os.chdir("..")


