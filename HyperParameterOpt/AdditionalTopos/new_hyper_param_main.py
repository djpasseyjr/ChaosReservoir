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
USER_ID = # ENTER USER #
BATCH_NUMBER = # ENTER BATCH NUMBER #

#edit directly into function for parameters,
generate_experiments(
    FNAME = USER_ID + str(BATCH_NUMBER),
    verbose = True,
    nets_per_experiment = 25,
    orbits_per_experiment = 1,
    num_experiments_per_file = 550,
    topology = # ENTER TOPOLOGY #,
    # these walltime parameters become the --time slurm command in bash_template
    hours_per_job = 72,
    # leave minutes_per_job at 0
    minutes_per_job = 0,
    # memory per job input is in Gigabytes
    memory_per_job = 4,

    # parameters below should be a list
    network_sizes = [500, 1500, 2500],
    # network_size = [None], #None means network size will be random between 2k-3.5k
    gamma_vals = [ 0.1, 0.5, 1, 2, 5, 10],
    sigma_vals = [.001, .005, .01, .05, .14, .2, 1],
    spectr_vals = [.1, .9, 1, 1.1, 2, 5, 10, 20],
    # topo_p_vals = [None], # Barabasi, loop, chain, ident and no edges
    # topo_p_vals = [.1, .5, .7] # Rewiring probability for Watts2 or Watts4 (We aren't doing other watts)
    # topo_p_vals = [.5, 1, 2, 3, 4, 5] # Mean degree for random digraph, erdos, and random geometric graphs
    ridge_alphas = [1.0, 0.01, 1e-4, 1e-6, 1e-8],
    # remove_p_list = [0] # For no edges
    # remove_p_list = [0, .1, .2, .3, .4, .5, .6, .7, .8, .86, .88 .9, .92, .94, .96, .98, .99] # For everything else
)
