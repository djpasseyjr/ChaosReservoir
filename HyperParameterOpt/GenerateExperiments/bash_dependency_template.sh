#!/bin/bash
echo "starting --dependency=singleton --job-name=#JOB_NAME#"
echo "\nstarting --dependency=singleton --job-name=#JOB_NAME#" >> my_saved_ids.txt
echo "$(date)" >> my_saved_ids.txt
echo "sbatch #FILENAME_PREFIX#.sh" >> my_saved_ids.txt
sbatch --dependency=singleton --job-name=#JOB_NAME# #FILENAME_PREFIX#.sh >> my_saved_ids.txt
echo "individual_partition_compilation_#FILENAME_PREFIX#.sh" >> my_saved_ids.txt
sbatch --dependency=singleton --job-name=#JOB_NAME# individual_partition_compilation_#FILENAME_PREFIX#.sh >> my_saved_ids.txt
echo "all_partitions_compilation_#FILENAME_PREFIX#.sh" >> my_saved_ids.txt
sbatch --dependency=singleton --job-name=#JOB_NAME# all_partitions_compilation_#FILENAME_PREFIX#.sh >> my_saved_ids.txt
echo "finished" >> my_saved_ids.txt

#how to include cleanup, and final.sh
# move the save_ids.txt file to a directory
# doing this with python could potentially write the job_id of the generation batch to the cleanup slurm
# start simple then amp it up

#or just have one file, where I add the date and time, and just append perpetually
