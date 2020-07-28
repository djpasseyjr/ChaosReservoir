#!/bin/bash

# not needed with supermain
# cp main.py #FILENAME_PREFIX#_main.py
# echo "copied main.py to #FILENAME_PREFIX#_main.py"

echo "starting --dependency=singleton --job-name=#JOB_NAME#"
echo "starting --dependency=singleton --job-name=#JOB_NAME#" >> my_saved_ids.txt
echo "$(date)" >> my_saved_ids.txt
echo "sbatch #FILENAME_PREFIX#.sh" >> my_saved_ids.txt
sbatch --dependency=singleton --job-name=#JOB_NAME# #FILENAME_PREFIX#.sh >> my_saved_ids.txt
echo "individual_partition_compilation_#FILENAME_PREFIX#.sh" >> my_saved_ids.txt
sbatch --dependency=singleton --job-name=#JOB_NAME# individual_partition_compilation_#FILENAME_PREFIX#.sh >> my_saved_ids.txt
echo "all_partitions_compilation_#FILENAME_PREFIX#.sh" >> my_saved_ids.txt
sbatch --dependency=singleton --job-name=#JOB_NAME# all_partitions_compilation_#FILENAME_PREFIX#.sh >> my_saved_ids.txt
echo "finished" >> my_saved_ids.txt
echo $"\n" >> my_saved_ids.txt
