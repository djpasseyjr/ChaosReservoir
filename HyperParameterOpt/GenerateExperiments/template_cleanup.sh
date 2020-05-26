#!/bin/bash
# this file is suppose to run immediately after submitting the batch

echo "note that once batch bash script finishes, all files (relevant to this batch) will be moved outside of ChaosReservoir to avoid the possibility of data being lost"
mkdir #DIR#/#FNAME#/

#move a copy of the batch bash script to batch directory
#cp #FNAME#.sh #DIR#/#FNAME#/copy_#FNAME#.sh


# when the jobs are initiated, slurm output files are generated,
# put all those slurm files into the output_slurm file
# assume that the topology directory has already been made, in default repository structure

mkdir #DIR#/#FNAME#/output_slurm
mv slurm* #DIR#/#FNAME#/output_slurm/

#make and move a copy of the main file, just in case, or for reference
cp main.py #DIR#/#FNAME#/main_#FNAME#.py
echo "once the cleanup_#FNAME#.sh file has been run, it can be deleted"
echo "rm -v cleanup_#FNAME#.sh"
