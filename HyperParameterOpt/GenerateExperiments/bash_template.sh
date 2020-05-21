#!/bin/bash
#SBATCH --time=#HOURS#:00:00              # walltime
#SBATCH --ntasks=1                        # number of processor cores (i.e. tasks)
#SBATCH --nodes=1                         # number of nodes, no need to change unless we use MPI
#SBATCH --mem-per-cpu=1024M                # memory per CPU core, 3072M = 3G, 512M = 0.5G
#SBATCH -J "#FNAME#"                      # job name
#SBATCH --array=0-#NUMBER_JOBS#           # the range is inclusive

module purge
module load python/3.7

python3 #DIR#/#FNAME#_${SLURM_ARRAY_TASK_ID}.py

# THE EFFICIENCY is the final sbatch argument --array, it's like a for loop #
# see the below resources
# https://rc.byu.edu/wiki/index.php?page=How+do+I+submit+a+large+number+of+very+similar+jobs%3F
# https://rc.byu.edu/wiki/?id=slurm-auto-array
# https://rc.byu.edu/wiki/index.php?page=SLURM+Tips+and+Tricks
