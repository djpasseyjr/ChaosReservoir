#!/bin/bash
#SBATCH --time=#HOURS#:00:00              # walltime
#SBATCH --ntasks=1                        # number of processor cores (i.e. tasks)
#SBATCH --nodes=1                         # number of nodes, no need to change unless we use MPI
#SBATCH --mem-per-cpu=1024M                # memory per CPU core, 3072M = 3G, 512M = 0.5G
#SBATCH -J "#FNAME#"                      # job name
#SBATCH --array=0-#NUMBER_JOBS#           # the range is inclusive

module purge
module load python/3.7

echo "note that the slurm files that will be created when the jobs run, will automatically be organized upon completion"
echo "most files will be located in the #DIR#/#FNAME#/ directory that will be created upon job completion"

python3 #DIR#/#FNAME#_${SLURM_ARRAY_TASK_ID}.py

# THE EFFICIENCY is the final sbatch argument --array, it's like a for loop #
# see the below resources
# https://rc.byu.edu/wiki/index.php?page=How+do+I+submit+a+large+number+of+very+similar+jobs%3F
# https://rc.byu.edu/wiki/?id=slurm-auto-array
# https://rc.byu.edu/wiki/index.php?page=SLURM+Tips+and+Tricks

# when the jobs are initiated, slurm output files are generated,
# put all those slurm files into the output_slurm file
# assume that the topology directory has already been made, in default repository structure 
mkdir #DIR#/#FNAME#/
mkdir #DIR#/#FNAME#/output_slurm
mv slurm* #DIR#/#FNAME#/output_slurm/

#make a copy of the main file, just in case
cp main.py main_#FNAME#.py
mv main_#FNAME#.py #DIR#/#FNAME#/

echo "after batch completes, run: bash cleanup_compile_#FNAME#.sh"
