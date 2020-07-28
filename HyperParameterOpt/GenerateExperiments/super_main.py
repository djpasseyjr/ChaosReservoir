from res_experiment import *
from rescomp.lorenz_sol import *
from parameter_experiments import *
from math import floor
from math import ceil
from scipy import integrate
import numpy as np

""" Here the user can input parameter ranges, and then parameter_experiments.py
will create individual experiment files, and bash scripts, to be run.
See the README.md for the file naming system for FNAME input
    in preparation for when we need to systematically
    read in each result pkl file exactly once (no more, no less) to one data source
    file per batch
Note that the supercomputer scheduler might delay our job because its sensitive to our estimate of
how long the each experiment will take to run. If the job runs longer than the projected walltime
then job is terminated
If you as the user would like emails about the job, then consider adding some lines
    to the specific bash_template (produced after running `python main.py`),
    see the slurm generator
    https://rc.byu.edu/documentation/slurm/script-generator
"""

#edit USER_ID, and BATCH_NUMBER to follow file naming style guide in README.md
#USER_ID should be one of the following: ['JW','DJ','BW','JJ','IB']
USER_ID = 'jw'
BATCH_NUMBER = 4

#edit directly into function for parameters,
verbose = True
# parameters for compilation
PARTITION_NUM = 1
compilation_hours_per_partition = 50
compilation_memory_per_partition = 50
#if bash2_desired is False, then bash2_walltime_hours, bash2_memory_required are irrelevant
bash2_desired=True
bash2_walltime_hours = 1
bash2_memory_required = 50
#essential parameters for experiment generation
nets_per_experiment = 25
orbits_per_experiment = 1
num_experiments_per_file = 1
topology = 'barab1'
# these walltime parameters become the --time slurm command in bash_template
minutes_per_experiment = 12
# leave minutes_per_job at 0
minutes_per_job = 0
# memory per job input is in Gigabytes
memory_per_job = 3

# parameters below should be a list of lists
network_sizes = [[500],[1500],[2500]]
# network_size = [None], #None means network size will be random between 2k-3.5k
gamma_vals = [[1]]
sigma_vals = [[0.01]]
spectr_vals = [[1]]
topo_p_vals = [[None]]
ridge_alphas = [[0.001]]
remove_p_list = [[0.1,0.5,0.8]]

super_bash_script = '#!/bin/bash\n\n'
batch_count = len(network_sizes) * len(gamma_vals) * len(sigma_vals) * len(spectr_vals) * len(topo_p_vals) * len(ridge_alphas) * len(remove_p_list)
print('number of batches',batch_count)

for a in network_sizes:
    for b in gamma_vals:
        for c in sigma_vals:
            for d in spectr_vals:
                for e in topo_p_vals:
                    for f in ridge_alphas:
                        for g in remove_p_list:
                            filename_prefix = USER_ID + str(BATCH_NUMBER) + "_" + topology
                            with open('main_template.py','r') as file:
                                tmpl_str = file.read()

                            print('\nbatch',filename_prefix)
                            number_of_experiments = len(a)*len(b)*len(c)*len(d)*len(e)*len(f)*len(g)
                            print('number_of_experiments',number_of_experiments)
                            exper_per = ceil(number_of_experiments / 1001)
                            print('input for experiments_per_file',exper_per)
                            print('estimated file count',ceil(number_of_experiments / exper_per))
                            #Calculate hours for each file
                            hours = ceil(minutes_per_experiment*exper_per/60)
                            #Write all main files
                            tmpl_str = tmpl_str.replace("#USERID#",USER_ID)
                            tmpl_str = tmpl_str.replace("#BATCH_NUM#",str(BATCH_NUMBER))
                            tmpl_str = tmpl_str.replace("#PARTITIONS#",str(PARTITION_NUM))
                            tmpl_str = tmpl_str.replace("#COMP_HOURS#",str(compilation_hours_per_partition))
                            tmpl_str = tmpl_str.replace("#COMP_MEM#",str(compilation_memory_per_partition))

                            tmpl_str = tmpl_str.replace("#BASH2#",str(bash2_desired))
                            tmpl_str = tmpl_str.replace("#BASH2_HOURS#",str(bash2_walltime_hours))
                            tmpl_str = tmpl_str.replace("#BASH2_MEM#",str(bash2_memory_required))

                            tmpl_str = tmpl_str.replace("#NETS_PER#",str(nets_per_experiment))
                            tmpl_str = tmpl_str.replace("#ORBITS_PER#",str(orbits_per_experiment))
                            tmpl_str = tmpl_str.replace("#EXPERIMENTS_PER#",str(exper_per))
                            tmpl_str = tmpl_str.replace("#TOPOLOGY#",str(topology))
                            tmpl_str = tmpl_str.replace("#HOURS#",str(hours))
                            tmpl_str = tmpl_str.replace("#MEMORY#",str(memory_per_job))

                            tmpl_str = tmpl_str.replace("#SIZES#",str(a))
                            tmpl_str = tmpl_str.replace("#GAMMAS#",str(b))
                            tmpl_str = tmpl_str.replace("#SIGMAS#",str(c))
                            tmpl_str = tmpl_str.replace("#SPECTRS#",str(d))
                            tmpl_str = tmpl_str.replace("#TOPO_PS#",str(e))
                            tmpl_str = tmpl_str.replace("#RIDGE_ALPHAS#",str(f))
                            tmpl_str = tmpl_str.replace("#REMOVE_PS#",str(g))


                            new_f = open(filename_prefix + '_main.py','w')
                            new_f.write(tmpl_str)
                            new_f.close()

                            super_bash_script += f"\npython {filename_prefix}_main.py\nbash run_{filename_prefix}.sh"

                            BATCH_NUMBER += 1

with open(f'super_bash_{filename_prefix}.sh','w') as sbIO:
    sbIO.write(super_bash_script)
print('finished')
