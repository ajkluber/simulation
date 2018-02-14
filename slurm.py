
def make_script_string(commands, jobname, gpus=False, partition="commons",
        walltime="23:55:00", email=False, emailtype="ALL", nodes=1,
        ntasks_per_node=1, mem_per_cpu=False, exclude_nodes=False,
        exclusive=False, dependency_type=False, dependency_ID=False,
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
    slurm +="#SBATCH --job-name={}\n".format(jobname)
    slurm +="#SBATCH --account=commons\n"
    if gpus:
        slurm +="#SBATCH --gres=gpu:{}\n".format(gpus)
    else:
        slurm +="#SBATCH --partition={}\n".format(partition)
    slurm +="#SBATCH --nodes={}\n".format(nodes)
    slurm +="#SBATCH --ntasks-per-node={}\n".format(ntasks_per_node)
    slurm +="#SBATCH --time={}\n".format(walltime)
    slurm +="#SBATCH --export=ALL\n"
    if exclude_nodes:
        slurm += "#SBATCH --exclude={}\n".format(exclude_nodes)
    if exclusive:
        slurm +="#SBATCH --exclusive\n"
    if mem_per_cpu:
        slurm +="#SBATCH --mem-per-cpu={}\n".format(mem_per_cpu)
    if email:
        slurm +="#SBATCH --mail-user={}\n".format(email)
        if emailtype:
            slurm +="#SBATCH --mail-type={}\n".format(emailtype)
    if dependency_type:
        slurm += "#SBATCH --dependency={}:{}\n".format(dependency_type, dependency_ID)
    if cd_slurm_dir:
        slurm +="cd $SLURM_SUBMIT_DIR\n"
    slurm += 'pwd\n'
    slurm += 'echo "JobID: $SLURM_JOB_ID"\n'
    slurm += 'echo "Running on: $SLURM_JOB_NODELIST"\n\n'
    slurm += "{}\n".format(commands)
    return slurm
