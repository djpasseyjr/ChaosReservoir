#!/bin/bash
#SBATCH --time=#HOURS#:#MINUTES#:00       # walltime
#SBATCH --ntasks=1                        # number of processor cores (i.e. tasks)
#SBATCH --nodes=1                         # number of nodes, no need to change unless we use MPI
#SBATCH --mem-per-cpu=#MEMORY#G           # memory per CPU core, 3072M = 3G, 1012M = 1G
#SBATCH -J "#FNAME#"                      # job name

# this file should be run once the batch completes

#compile the output
python compile_output_#FNAME#.py
