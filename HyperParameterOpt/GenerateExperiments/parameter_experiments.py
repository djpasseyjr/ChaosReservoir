import networkx as nx
import numpy as np
import pickle
from math import floor, ceil
from rescomp import ResComp, specialize, lorenz_equ
from res_experiment import *
from scipy import sparse

# parameter used in write_bash_script, this is in minutes
WALLTIME_PER_JOB = 300

def prepare_output_compilation(directory,filename, number_of_experiments):
    """
    write the directory and number_of_experiments to the compile_output.py file

    Record the topology so that the ouput_compiler can look in that
    directory to compile all the pkl files that store the output
    from the experiments

    Parameters:
        directory               (str): the name of output directory where all resulting pkl files will be stored
        filename                (str): the filename prefix that all the files have in common
        number_of_experiments   (int): the number of experiments is used to systematically
                                        compile all individual output files into one primary file
    """
    tmpl_stream = open('compile_output.py','r')
    tmpl_str = tmpl_stream.read()
    tmpl_str = tmpl_str.replace("#TOPOLOGY_DIRECTORY#",directory)
    tmpl_str = tmpl_str.replace("#FNAME#",filename)
    tmpl_str = tmpl_str.replace("#NUMBER_OF_EXPERIMENTS#",str(number_of_experiments))
    new_name = 'compile_output_' + filename +'.py'
    new_f = open(new_name,'w')
    new_f.write(tmpl_str)
    new_f.close()
    print(f'\nOnce output has been produced, run the command below:\npython {new_name}')

def directory(network):
    """
    Given a certain topology, output the string of the
        directory where all the modified experiment_template.py files
        should be saved

    Parameters:
        Network (str): The topology for the experiments

    Returns:
        DIR (str): The directory where the individual experiment.py files will be stored
    """
    # the network options here should match the generate_adj function in res_experiment.py
    network_options = ['barab1', 'barab2', 'erdos', 'random_digraph', 'watts3', 'watts5','geom']
    if network not in network_options:
        raise ValueError('{network} not in {network_options}')

    if network == 'barab1' or network == 'barab2':
        DIR = 'Barabasi'
    if network == 'erdos':
        DIR = 'Erdos'
    if network == 'random_digraph':
        DIR = 'RandDigraph'
    if network == 'watts3' or network == 'watts5':
        DIR = 'Watts'
    if network == 'geom':
        DIR = 'Geometric'
    return DIR

def write_bash_script(directory,filename, number_of_experiments):
    """
    Write the bash script to run all the experiments, for reasoning
    behind this format, see the links in the bash_template.sh file

    Parameters:
        directory               (str): the name of output directory where all resulting pkl files will be stored
        filename                (str): the filename prefix that all the files have in common
        number_of_experiments   (int): the number of experiments is used to systematically
                                        compile all individual output files into one primary file
    """
    # WALLTIME_PER_JOB is in minutes, this may depend upon topology size
    # WALLTIME_PER_JOB = 30
    # find the number of hours, then round up to next hour
    TOTAL_TIME = ceil(WALLTIME_PER_JOB  / 60) # this assumes same number of processors as experiments
    # TOTAL_TIME = ceil(WALLTIME_PER_JOB * number_of_experiments / 60) #this is if only one processor


    tmpl_stream = open('bash_template.sh','r')
    tmpl_str = tmpl_stream.read()
    tmpl_str = tmpl_str.replace("#HOURS#",str(TOTAL_TIME))
    tmpl_str = tmpl_str.replace("#DIR#",directory)
    tmpl_str = tmpl_str.replace("#FNAME#",filename)
    # we want a processor for each experiment
    tmpl_str = tmpl_str.replace("#CORES#",str(number_of_experiments))
    #subtract the number of experiments by one because of zero indexing of filenames
    # whereas the slurm --array range is inclusive on endpoints
    # for example, see https://rc.byu.edu/wiki/index.php?page=How+do+I+submit+a+large+number+of+very+similar+jobs%3F
    #       then search "	Resulting task ID's	" on that webpage to see 0-6 is inclusive on endpoints
    tmpl_str = tmpl_str.replace("#NUMBER_JOBS#",str(number_of_experiments - 1))
    new_f = open(filename +'.sh','w')
    new_f.write(tmpl_str)
    new_f.close()
    print('NEXT: sbatch',filename +'.sh')

def generate_experiments(
    FNAME,
    nets_per_experiment = 2,
    orbits_per_experiment = 200,
    topology = None,
    network_sizes = [2000],
    gamma_vals = [1],
    sigma_vals = [0.3],
    spectr_vals = [0.9],
    topo_p_vals = [None],
    ridge_alphas = [0.001],
    remove_p_list = [0.5]
):
    """ Write one bash file (according to bash_template.sh), and individual
    experiment files (according to experiment_template.py) for a grid of
    parameters ranges. See the README for a style guide for fname.

    Parameters:
        FNAME                       (str):   prefix to each filename, an error will be thrown if not specified
        nets_per_experiment         (int):   number of networks to generate for a given topology
        orbits_per_experiment       (int):   number of orbits to run on each network for a given topology
        topology                    (str):   topology as specified in the generate_adj function of res_experiment.py, an error will be thrown if not specified
        network_sizes               (list):  sizes for the network topologies
        gamma_vals                  (list):  gamma values for reservoir
        sigma_vals                  (list):  sigma values for reservoir
        spectr_vals                 (list):  spectral radius values for reservoir
        topo_p_vals                 (list):  may not be Necessary for certain topologies
        ridge_alphas                (list):  ridge alpha values for reservoir for regularization of the model
        remove_p_list               (list):  the percentages of edges in the adjacency matrix to remove

    Returns:
        None
    Output:
        Writes files to other directories

    """
    if topology is None:
        raise ValueError('Please Specify a Topology as specified in the generate_adj function of res_experiment.py')

    # in order to separate different topology's .py files into directories
    # then find the directory for this specific topology
    DIR = directory(topology)

    # the counter will be the final component of each file name, it's an enumeration of all the parameters
    # the parameters values are not in the filename
    parameter_enumaration_number = 0
    for n in network_sizes:
        for TOPO_P in topo_p_vals:
            for gamma in gamma_vals:
                for sigma in sigma_vals:
                    for spectr in spectr_vals:
                        for ridge_alpha in ridge_alphas:
                            for p in remove_p_list:

                                #put together FNAME with topology, and parameter_enumaration_number
                                save_fname =  DIR + '/' + FNAME + "_" + topology + "_" + str(parameter_enumaration_number)

                                #read in template experiment file
                                tmpl_stream = open('experiment_template.py','r')
                                tmpl_str = tmpl_stream.read()
                                tmpl_str = tmpl_str.replace("#FNAME#",save_fname + '.pkl')
                                tmpl_str = tmpl_str.replace("#TOPOLOGY#",topology)
                                tmpl_str = tmpl_str.replace("#TOPO_P#",str(TOPO_P))
                                tmpl_str = tmpl_str.replace("#REMOVE_P#",str(p))
                                tmpl_str = tmpl_str.replace("#RIDGE_ALPHA#",str(ridge_alpha))
                                tmpl_str = tmpl_str.replace("#SPECT_RAD#",str(spectr))
                                tmpl_str = tmpl_str.replace("#GAMMA#",str(gamma))
                                tmpl_str = tmpl_str.replace("#SIGMA#",str(sigma))
                                tmpl_str = tmpl_str.replace("#NETS_PER_EXPERIMENT#",str(nets_per_experiment))
                                tmpl_str = tmpl_str.replace("#ORBITS_PER_EXPERIMENT#",str(orbits_per_experiment))
                                tmpl_str = tmpl_str.replace("#SIZE_OF_NETWORK#",str(n))
                                # Save to new file
                                new_f = open(save_fname + '.py','w')
                                new_f.write(tmpl_str)
                                new_f.close()

                                parameter_enumaration_number += 1

    print('\ntotal number of experiments:',parameter_enumaration_number)
    #in order to run all the experiments on the supercomputer we need the main bash script
    write_bash_script(DIR,FNAME + "_" + topology,parameter_enumaration_number)
    #in order to compile output systematically, store the number of experiments and output directory
    prepare_output_compilation(DIR,FNAME + "_" + topology,parameter_enumaration_number)
