
def make_script_string(commands, jobname, gpus=False, partition="commons", 
        walltime="23:55:00", email=False, emailtype="ae", nodes=1, 
        ntasks_per_node=1, mem_per_cpu=False, exclusive=False,
        cd_slurm_dir=True):
    """Return slurm script string that can be written to file
    
    Parameters
    ----------
    commands : str
        commnands to be executed 
    jobname : str
    
    gpus : bool, opt.

    partition :

    Returns
    -------
    
    """

    slurm = "#!/bin/bash\n"
    slurm +="#SBATCH --job-name=%s\n" % jobname
    if gpus:
        slurm +="#SBATCH --gres=gpu:%d\n" % gpus
    else:
        slurm +="#SBATCH --partition=%s\n" % partition
    slurm +="#SBATCH --nodes=%d\n"% nodes
    slurm +="#SBATCH --ntasks-per-node=%d\n" % ntasks_per_node
    slurm +="#SBATCH --time=%s\n" % walltime
    slurm +="#SBATCH --export=ALL\n"
    if exclusive:
        slurm_string +="#SBATCH --exclusive\n"
    if mem_per_cpu:
        slurm +="#SBATCH --mem-per-cpu=%s\n" % mem_per_cpu
    if email:
        slurm_string +="#SBATCH --mail-user=%s\n" % email
        if emailtype:
            slurm_string +="#SBATCH --mail-type=%s\n" % emailtype 
    if cd_slurm_dir:
        slurm +="cd $SLURM_SUBMIT_DIR\n\n"
    else:
        slurm +="\n"
    slurm += commands
    return slurm

