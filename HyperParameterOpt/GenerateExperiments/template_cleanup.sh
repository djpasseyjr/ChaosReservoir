#!/bin/bash
# this file is suppose to run immediately after submitting the batch

echo "note that once batch bash script finishes, all files (relevant to this batch) will be moved outside of ChaosReservoir to avoid the possibility of data being lost"
mkdir #DIR#/#FNAME#/

#move the bash script to batch directory
mv #FNAME#.sh #DIR#/#FNAME#/


# when the jobs are initiated, slurm output files are generated,
# put all those slurm files into the output_slurm file
# assume that the topology directory has already been made, in default repository structure

mkdir #DIR#/#FNAME#/output_slurm
mv slurm* #DIR#/#FNAME#/output_slurm/

#make a copy of the main file, just in case
cp main.py main_#FNAME#.py
mv main_#FNAME#.py #DIR#/#FNAME#/
echo "once the cleanup_compile_#FNAME#.sh file has been run, it can be deleted"
echo "rm -v cleanup_#FNAME#.sh"
