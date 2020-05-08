import networkx as nx
import numpy as np
import pickle
from math import floor
from rescomp import ResComp, specialize, lorenz_equ
from res_experiment import *
from scipy import sparse

"""
to avoid having to make tons of experiment files and bash files for each
experiment, just have one bash file to run this file, so copy over some code from template.py
"""

def condense_output():
    """ """
    # unless I can run a command that will delete files, then I probably don't want to write to many files, just one
    # join together many dictionaries,
    raise NotImplementedError('condense_output not finished')

def parameter_dictionaries(RIDGE_ALPHA,SPECTR,GAMMA,SIGMA):
    """ """
    DIFF_EQ_PARAMS = {
                      "x0": [-20, 10, -.5],
                      "begin": 0,
                      "end": 60,
                      "timesteps":60000,
                      "train_per": .66,
                      "solver": lorenz_equ
                     }

    RES_PARAMS = {
                  "uniform_weights": True,
                  "solver": "ridge",
                  "ridge_alpha": RIDGE_ALPHA,
                  "signal_dim": 3,
                  "network": "random graph",

                  "res_sz": 15,
                  "activ_f": np.tanh,
                  "connect_p": .4,
                  "spect_rad": SPECTR,
                  "gamma": GAMMA,
                  "sigma": SIGMA,
                  "sparse_res": True,
                 }
    return DIFF_EQ_PARAMS, RES_PARAMS

def generate_experiments(
    # save_file_name,
    # total_nets, #not really sure what this is for since number
    # orbits_per_net,
    condensed_output_filename = None,
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
    """ Run a grid of experiments based upon parameters

    condensed_output_filename   (str):   if None, then individual files will be generated per experiment, else input file name
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
    Outputs: A pkl file with all the results from each experiment

    """
    if condensed_output_filename is None:
        message = """ Currently generate_experiments doesn\'t include funtionality
            for generating multiple files for each experimient """
        raise NotImplementedError(message)

    if topology is None:
        raise ValueError('Please Specify a Topology as specified in the generate_adj function of res_experiment.py')

    for TOPO_P in topo_p_vals:
        for gamma in gamma_vals:
            for sigma in sigma_vals:
                for spectr in spectr_vals:
                    for specific_ridge_alpha in ridge_alphas:
                        for p in remove_p_list:
                            # run experiment
                            DIFF_EQ_PARAMS, RES_PARAMS = parameter_dictionaries(
                                    specific_ridge_alpha, spectr, gamma, sigma)

                            #fname isn't necessary in the case that we only write to one big file
                            #random_lorenz_x0 comes from res_experiment.py
                            results = experiment(
                                                'test',
                                                topology,
                                                TOPO_P,
                                                RES_PARAMS,
                                                DIFF_EQ_PARAMS,
                                                ntrials=nets_per_experiment,
                                                norbits=orbits_per_experiment,
                                                x0=random_lorenz_x0,
                                                remove_p=p
                                            )
                            print(results,'\n\n')
                            print('figure out how to take the results and write to one big file')
                            print('remove return statement, its temporary just avoid loop overprocessing')
                            return




def test_functions():
    """ """
    try:
        generate_experiments()
    except:
        print('writing individual files still not built for generate_experiments\n')

    a,b = parameter_dictionaries(.0001,1,2,3)
    print(b)
    assert b == {
                  "uniform_weights": True,
                  "solver": "ridge",
                  "ridge_alpha": .0001,
                  "signal_dim": 3,
                  "network": "random graph",

                  "res_sz": 15,
                  "activ_f": np.tanh,
                  "connect_p": .4,
                  "spect_rad": 1,
                  "gamma": 2,
                  "sigma": 3,
                  "sparse_res": True,
                 } , 'failed test case for parameter_dictionaries'

    generate_experiments('joeytest101','barab1')



# test_functions()
message = 'create a sample function to generate a sample experiment.py file and sample experiment.sh file, if im using individual py and sh files'
print(message)
