import networkx as nx
import numpy as np
import pickle
from math import floor
from rescomp import ResComp, specialize, lorenz_equ
from res_experiment import *
from scipy import sparse

""" See the README.md for the file naming system for FNAME input

in preparation for when we need to systematically
read in each result pkl file exactly once (no more, no less) to one data source
file per batch,

this parameters_experiments file is just to create the
needed files to run experiments. Reading the results will be a different process
"""

def generate_experiments(
    fname,
    nets_per_experiment = 5,
    orbits_per_experiment = 5,
    topology = None,
    gamma_vals = [1],
    sigma_vals = [1],
    spectr_vals = [0.9],
    topo_p_vals = [None],
    ridge_alphas = [0.001],
    remove_p_list = [0]
):
    """ Write individual bash files (according to bash_template.sh) and
    experiment files (according to experiment_template.py) for a grid of
    parameters ranges. See the README for a style guide for fname.

    fname                       (str):   prefix to each filename, an error will be thrown if not specified
    nets_per_experiment         (int):   number of networks to generate for a given topology
    orbits_per_experiment       (int):   number of orbits to run on each network for a given topology
    topology                    (str):   topology as specified in the generate_adj function of res_experiment.py
    gamma_vals                  (list):  gamma values for reservoir
    sigma_vals                  (list):  sigma values for reservoir
    spectr_vals                 (list):  spectral radius values for reservoir
    topo_p_vals                 (list):  may not be Necessary for certain topologies
    ridge_alphas                (list):  ridge alpha values for reservoir for regularization of the model
    remove_p_list               (list):  the percentages of edges in the adjacency matrix to remove

    Returns: None

    """
    if topology is None:
        raise ValueError('Please Specify a Topology as specified in the generate_adj function of res_experiment.py')

    # the counter will be the final component of each file, it's an enumeration of all the parameters
    # the parameters themselves are not in the filename
    parameter_enumaration_number = 1
    for TOPO_P in topo_p_vals:
        for gamma in gamma_vals:
            for sigma in sigma_vals:
                for spectr in spectr_vals:
                    for ridge_alpha in ridge_alphas:
                        for p in remove_p_list:
                            #put together FNAME with topology, and parameter_enumaration_number
                            save_fname = FNAME + "_" + topology + "_" + parameter_enumaration_number

                            #read in template experiment file
                            tmpl_stream = open('experiment_template.py','r')
                            tmpl_str = tmpl_stream.read()
                            tmpl_str = tmpl_str.replace("#FNAME#",save_fname + '.pkl')
                            tmpl_str = tmpl_str.replace("#TOPOLOGY#",toplogy)
                            tmpl_str = tmpl_str.replace("#TOPO_P#",TOPO_P)
                            tmpl_str = tmpl_str.replace("#REMOVE_P#",p)
                            tmpl_str = tmpl_str.replace("#RIDGE_ALPHA#",ridge_alpha)
                            tmpl_str = tmpl_str.replace("#SPECT_RAD#",spectr)
                            tmpl_str = tmpl_str.replace("#GAMMA#",gamma)
                            tmpl_str = tmpl_str.replace("#SIGMA#",sigma)
                            # Save to new file
                            new_f = open(save_fname + '.py','w')
                            new_f.write(tmpl_str)
                            new_f.close()

                            #write bash file
                            # can one bash rile run all the experiment files,
                            # or does each each experiment file need it's own bash file? 

                            parameter_enumaration_number += 1
                            return
