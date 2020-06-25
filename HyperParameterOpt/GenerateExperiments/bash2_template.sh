#!/bin/bash
#SBATCH --time=#HOURS#:00:00       # walltime
#SBATCH --ntasks=1                        # number of processor cores (i.e. tasks)
#SBATCH --nodes=1                         # number of nodes, no need to change unless we use MPI
#SBATCH --mem-per-cpu=#MEMORY#G           # memory per CPU core, 3072M = 3G, 1012M = 1G
#SBATCH -J "#JNAME#"                      # job name
#SBATCH --array=0-#NUMBER_JOBS#           # the range is inclusive

module purge
module load python/3.7

# these sbatch commands have no utility unless they are placed before the first non-slurm commands
# so if you want emails then cut and paste these lines above `module purge`, as well as update email
#SBATCH --mail-user=example@byu.edu   # email address, change email
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

python3 #FNAME#_${SLURM_ARRAY_TASK_ID}.py
