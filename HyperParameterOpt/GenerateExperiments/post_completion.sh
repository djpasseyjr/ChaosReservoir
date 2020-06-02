#!/bin/bash
# this file should be run once the batch completes

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

# move the batch directory a better location

cd #DIR#
# the data can easily, automatically be moved to a folder in a file sharing location.
# mv #FNAME#/ ~/fsl_groups/fslg_webb_reservoir/compute/jw_data/jw38+
# mv #FNAME#/ ~/fsl_groups/fslg_webb_reservoir/dj_data
# mv #FNAME#/ ~/fsl_groups/fslg_webb_reservoir/jj_data
mv #FNAME#/ ~/fsl_groups/fslg_webb_reservoir/compute/unsorted_data
# mv #FNAME#/ ~/compute/Saved_data/
# mv #FNAME#/ ~/compute
cd ..

#any echo statements in this file, are "output" in each slurm file of the batch
