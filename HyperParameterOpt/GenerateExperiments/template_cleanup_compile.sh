#!/bin/bash
# this file is suppose to run once the batch has completed

#move the bash script to batch directory
mv #FNAME#.sh #DIR#/#FNAME#

#compile the output
python compile_output_#FNAME#.py

#move the compiled output
mv compiled_output_#FNAME#.pkl #DIR#/#FNAME#/

#move the compile output file to batch directory
mv compile_output_#FNAME#.py #DIR#/#FNAME#

mkdir #DIR#/#FNAME#/experiment_files
mkdir #DIR#/#FNAME#/result_files

#organize the .py files,
mv #DIR#/#FNAME#*.py #DIR#/#FNAME#/experiment_files
#organize the .pkl files
mv #DIR#/#FNAME#*.pkl #DIR#/#FNAME#/result_files

# move the batch directory to saved_data or to compute directory
# mkdir ~/compute/Saved_data/#FNAME#
# mv #DIR#/#FNAME#/ ~/compute/Saved_data/#FNAME#
mkdir ~/compute/#FNAME#
mv #DIR#/#FNAME#/ ~/compute/#FNAME#

echo "once the cleanup_compile_#FNAME#.sh file has been run, it can be deleted"
echo "rm -r -v cleanup_compile_#FNAME#.sh"
