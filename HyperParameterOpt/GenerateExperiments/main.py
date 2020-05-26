from res_experiment import *
from rescomp.lorenz_sol import *
from parameter_experiments import *
from math import floor
from scipy import integrate

""" Here the user can input parameter ranges, and then parameter_experiments.py
will create individual experiment files, and bash scripts, to be run.

See the README.md for the file naming system for FNAME input
    in preparation for when we need to systematically
    read in each result pkl file exactly once (no more, no less) to one data source
    file per batch

Note that the supercomputer scheduler might delay our job because its sensitive to our estimate of
how long the each experiment will take to run.

***     It is recommended that the WALLTIME_PER_JOB variable
in parameter_experiments be edited based upon size of network, etc ***

If you as the user would like emails about the job, then consider adding some lines
    to the specific bash_template (produced after running `python main.py`),
    see the slurm generator
    https://rc.byu.edu/documentation/slurm/script-generator
"""

#edit USER_ID, and BATCH_NUMBER to follow file naming style guide in README.md
#USER_ID should be one of the following: ['JW','DJ','BW','JJ','IB']
USER_ID = 'JW'
BATCH_NUMBER = 4

#edit directly into function for parameters,
generate_experiments(
    FNAME = USER_ID + str(BATCH_NUMBER),
    nets_per_experiment = 2,
    orbits_per_experiment = 200,
    topology = 'barab1',
    # these walltime parameters become the --time slurm command in bash_template
    hours_per_job = 5,
    # leave minutes_per_job at 0
    minutes_per_job = 0,
    # memory per job input is in Gigabytes
    memory_per_job = 3,

    # parameters below should be a list
    network_sizes = [3000],
    # network_size = [None], #None means network size will be random between 2k-3.5k
    gamma_vals = [1],
    sigma_vals = [0.01],
    spectr_vals = [1],
    topo_p_vals = [None],
    ridge_alphas = [0.001],
    remove_p_list = [0.1,0.5,0.8]
)
