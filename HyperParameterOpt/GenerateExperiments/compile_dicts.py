#compile_dicts.py
import numpy as np
import pickle
import pandas as pd
import time

DIR = "#TOPOLOGY_DIRECTORY#"
filename_prefix = "#FNAME#"
NEXPERIMENTS = #NUMBER_OF_EXPERIMENTS#
NETS_PER_EXPERIMENT = #NETS_PER_EXPERIMENT#
#verbose will become a parameter in main
verbose = #VERBOSE#

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

def compile_output(DIR, filename_prefix, num_experiments, nets_per_experiment):
    """
    Compile the data from all the various pkl files
    Parameters:
        DIR                     (str): The directory where the individual experiment.py files will be stored
        filename_prefix         (str): prefix to each filename, an error will be thrown if not specified
        num_experiments         (int): number of experiments total
        nets_per_experiment     (int): number of nets in each experiment, equivalent to nets_per_experiment in main.py
    """
    # Make dictionary for storing all data
    compiled = empty_result_dict(num_experiments, nets_per_experiment)

    # we also need the prefix of the files, or can we use os.listdir()
    # path is probably directory plus filename prefix
    path = DIR + "/" + filename_prefix + "_"
    failed_file_count = 0
    start = time.time()
    start_idx = 0

    if verbose:
        file = filename_prefix
        timing = '\n\n'

    for i in range(1, num_experiments+1):
        # Load next data dictionary
        try:
            data_dict = pickle.load(open(path + str(i) + '.pkl','rb'))
            # Add data to compiled dictionary
            add_to_compiled(compiled, data_dict, start_idx)
        except:
            failed_file_count += 1
        # Track experiment number
        for k in range(start_idx, start_idx + nets_per_experiment):
            compiled["exp_num"][k] = i

        start_idx += nets_per_experiment

        if verbose:
            if i % 1000 == 0:
                info = f'\n{i} files compile attempted,time since start (minutes):{round((time.time() - start )/ 60,1)}'
                timing += info
                print(info)
    #write final dict to pkl file
    pickle.dump(compiled, open('compiled_output_' + filename_prefix + '.pkl', 'wb'))

    if verbose:
        # Time difference is originally seconds
        finished = (time.time() - start )/ 60
        info = f'\nit took {round(finished,1)} minutes to compile\nor {round(finished / 60,1)} hours'
        file += info
        print(info)
        info = f'\n(#failed files) / (# total number of experiments) is {failed_file_count} / {NEXPERIMENTS}\nor {100 * round(failed_file_count/NEXPERIMENTS, 1)}% failed'
        file += info
        print(info)
        ending = f'\n{filename_prefix} compilation process finished'
        timing += ending
        #only write to the file once, the file will close automatically
        with open(f'{filename_prefix}_compiling_notes.txt','w') as f:
            f.write(file + timing)


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
    empty["exp_num"] = [-1] * nentries
    return empty

def add_to_compiled(compiled, data_dict, start_idx):
    """ Add output dictionary to compiled data, return next empty index """
    for k in data_dict.keys():
        for colname in FLOAT_COLNAMES + STRING_COLNAMES + LIST_COLNAMES:
            compiled[colname][start_idx + k] = data_dict[k][colname]

def merge_compiled(compiled1, compiled2):
    """ Merge two compiled dictionaries """
    if isinstance(compiled1, str) and isinstance(compiled2, str):
        compiled1 = pickle.load(open(compiled1, 'rb'))
        compiled2 = pickle.load(open(compiled2, 'rb'))
    # Shift experiment number for compiled2
    total_exp = np.max(compiled1["exp_num"])
    exp_nums = np.array(compiled2["exp_num"])
    exp_nums[exp_nums >= 0] += total_exp 
    compiled2["exp_num"] = list(exp_nums)
    # Merge
    for k in compiled1.keys():
        compiled1[k] += compiled2[k]
    return compiled1



compile_output(DIR,filename_prefix,NEXPERIMENTS,NETS_PER_EXPERIMENT)
