#!/bin/bash
#SBATCH --time=#HOURS#:#MINUTES#:00       # walltime
#SBATCH --ntasks=1                        # number of processor cores (i.e. tasks)
#SBATCH --nodes=1                         # number of nodes, no need to change unless we use MPI
#SBATCH --mem-per-cpu=#MEMORY#G           # memory per CPU core, 3072M = 3G, 1012M = 1G
#SBATCH -J "#FNAME#"                      # job name
#SBATCH --array=0-#NUMBER_JOBS#           # the range is inclusive

module purge
module load python/3.7

# these sbatch commands have no utility unless they are placed before the first non-slurm commands
# so if you want emails then cut and paste these lines above `module purge`, as well as update email
#SBATCH --mail-user=example@byu.edu   # email address, change email
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

python3 #DIR#/#FNAME#_${SLURM_ARRAY_TASK_ID}.py

#compile the output
python compile_output_#FNAME#.py

#move the compiled output
mv compiled_output_#FNAME#.pkl #DIR#/#FNAME#/

#move the compile output file to batch directory
mv compile_output_#FNAME#.py #DIR#/#FNAME#

# make directories for organization
mkdir #DIR#/#FNAME#/experiment_files
mkdir #DIR#/#FNAME#/result_files
#organize the .py files,
mv #DIR#/#FNAME#*.py #DIR#/#FNAME#/experiment_files
#organize the .pkl files
mv #DIR#/#FNAME#*.pkl #DIR#/#FNAME#/result_files

# move the batch directory to saved_data or to compute directory

cd #DIR#
# mv #FNAME#/ ~/compute/Saved_data/
mv #FNAME#/ ~/compute
cd ..

#any echo statements in this file, are "output" in each slurm file of the batch
