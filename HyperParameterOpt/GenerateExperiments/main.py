from res_experiment import *
from rescomp.lorenz_sol import *
from parameter_experiments import *
from math import floor
from scipy import integrate

""" there should only be one file that the user will have to edit,
so here the user can input parameter ranges, and then parameter_experiments.py
will create individual experiment files, and bash scripts, to be run,

Next part will figure out how to know when all the experiments have been run,
and then compile all the pkl files into one results file, that we can combine as
well with other results from other users.  

"""

#edit directly into function
generate_experiments(
    condensed_output_filename = 'filename',
    nets_per_experiment = 2,
    orbits_per_experiment = 2,
    topology = 'barab1',
    # parameters below should be a list
    gamma_vals = [1],
    sigma_vals = [1],
    spectr_vals = [0.9],
    topo_p_vals = [None],
    ridge_alphas = [0.001],
    remove_p_list = [0]
)
