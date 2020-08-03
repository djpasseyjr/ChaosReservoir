from res_experiment import *
from rescomp.lorenz_sol import *
from parameter_experiments import *
from math import floor
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
USER_ID = '#USERID#'
BATCH_NUMBER = #BATCH_NUM#

#edit directly into function for parameters,
generate_experiments(
    FNAME = USER_ID + str(BATCH_NUMBER),
    verbose = True,
    # parameters for compilation
    PARTITION_NUM = #PARTITIONS#,
    compilation_hours_per_partition = #COMP_HOURS#,
    compilation_memory_per_partition = #COMP_MEM#,
    #if bash2_desired is False, then bash2_walltime_hours, bash2_memory_required are irrelevant
    bash2_desired=#BASH2#,
    bash2_walltime_hours = #BASH2_HOURS#,
    bash2_memory_required = #BASH2_MEM#,
    #essential parameters for experiment generation
    nets_per_experiment = #NETS_PER#,
    orbits_per_experiment = #ORBITS_PER#,
    num_experiments_per_file = #EXPERIMENTS_PER#,
    topology = '#TOPOLOGY#',
    # these walltime parameters become the --time slurm command in bash_template
    hours_per_job = #HOURS#,
    # leave minutes_per_job at 0
    minutes_per_job = 0,
    # memory per job input is in Gigabytes
    memory_per_job = #MEMORY#,

    # parameters below should be a list
    network_sizes = #SIZES#,
    # network_size = [None], #None means network size will be random between 2k-3.5k
    gamma_vals = #GAMMAS#,
    sigma_vals = #SIGMAS#,
    spectr_vals = #SPECTRS#,
    topo_p_vals = #TOPO_PS#,
    ridge_alphas = #RIDGE_ALPHAS#,
    remove_p_list = #REMOVE_PS#
)