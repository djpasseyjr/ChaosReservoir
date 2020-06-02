import numpy as np
import pickle
import pandas as pd
import time


FLOAT_COLNAMES = [
    "mean_pred",
    "mean_err",
    "adj_size",
    "topo_p",
    "gamma",
    "sigma",
    "spect_rad",
    "ridge_alpha",
    "remove_p"
]
LIST_COLNAMES = [
    "pred",
    "err"
]
STRING_COLNAMES = [
    "net"
]
DIR = "#TOPOLOGY_DIRECTORY#"
filename_prefix = "#FNAME#"
NEXPERIMENTS = #NUMBER_OF_EXPERIMENTS#
NETS_PER_EXPERIMENT = #NETS_PER_EXPERIMENT#
#verbose will become a parameter in main
verbose = True



def compile_output(DIR, filename_prefix, num_experiments, nets_per_experiment):
    """
    Compile the data from all the various pkl files

    Parameters:
        DIR                     (str): The directory where the individual experiment.py files will be stored
        filename_prefix         (str): prefix to each filename, an error will be thrown if not specified
        total_experiment_number (int): the total number of experiment files that were created
                                       as described by the final parameter_enumaration_number in
                                       the generate_experiments() function of the parameter_experiments.py file
    """
    # Make dictionary for storing all data
    compiled = empty_result_dict(num_experiments, nets_per_experiment)

    # we also need the prefix of the files, or can we use os.listdir()
    # path is probably directory plus filename prefix
    path = DIR + "/" + filename_prefix + "_"
    failed_file_count = 0
    start = time.time()
    start_idx = 0

    for i in range(1, num_experiments+1):
        # Load next data dictionary
        try:
            data_dict = pickle.load(open(path + str(i) + '.pkl','rb'))
        except:
            failed_file_count += 1
        # Add data to compiled dictionary
        add_to_compiled(compiled, data_dict, start_idx)
        start_idx += nets_per_experiment
        if verbose:
            if i % 1000 == 0:
                print(f'{i} files compile attempted,\ntime since start (minues):{round((time.time() - start )/ 60,1)}')
    if verbose:
        # Time difference is originally seconds
        finished = (time.time() - start )/ 60
        print(f'it took {round(finished,1)} minutes to compile\nor {round(finished / 60,1)} hours')
        print(f'(#failed files) / (# total number of experiments) is {failed_file_count} / {NEXPERIMENTS}\nor {100 * round(failed_file_count/total_experiment_number,1)}% failed')
    #write final dict to pkl file
    pickle.dump(compiled, open('compiled_output_' + filename_prefix + '.pkl', 'wb'))
    if verbose:
        print(f'{filename_prefix} compilation process finished')

def empty_result_dict(num_experiments, nets_per_experiment):
    """ Make empty dictionary for compiling data """
    empty = {}
    nentries = num_experiments * nets_per_experiment
    for colname in FLOAT_COLNAMES:
        empty[colname] = [0.0] * nentries
    for colname in LIST_COLNAMES:
        empty[colname] = [[]] * nentries
    for colname in STRING_COLNAMES:
        empty[colname] = [''] * nentries
    return empty

def add_to_compiled(compiled, data_dict, start_idx):
    """ Add output dictionary to compiled data, return next empty index """
    for k in data_dict.keys():
        for colname in FLOAT_COLNAMES + STRING_COLNAMES + LIST_COLNAMES:
            compiled[colname][start_idx + k] = data_dict[k][colname]
