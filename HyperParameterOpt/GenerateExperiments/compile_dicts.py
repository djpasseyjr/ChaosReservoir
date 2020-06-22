#compile_dicts.py
import numpy as np
import pickle
import pandas as pd
import time

DIR = "#TOPOLOGY_DIRECTORY#"
filename_prefix = "#FNAME#"
NEXPERIMENTS = #NUMBER_OF_EXPERIMENTS#
NETS_PER_EXPERIMENT = #NETS_PER_EXPERIMENT#
num_experiments_per_file = #NUM_EXPRMTS_PER_FILE#
#verbose will become a parameter in main
verbose = #VERBOSE#

COLNAMES = [
    "mean_pred",
    "mean_err",
    "adj_size",
    "topo_p",
    "gamma",
    "sigma",
    "spect_rad",
    "ridge_alpha",
    "remove_p",
    "pred",
    "err",
]

NETCOLS = [
    "max_scc",
    "max_wcc",
    "giant_comp",
    "singletons",
    "nwcc",
    "nscc"
    "cluster",
    "assort",
    "diam"
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
    #get a list of failed files identifiers so it's simple to check traceback
    failed_experiment_identifiers = []
    failed_job_identifiers = []
    
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

    for i in range(num_experiments):
        # Load next data dictionary
        try:
            data_dict = pickle.load(open(path + str(i) + '.pkl','rb'))
            # Add data to compiled dictionary
            add_to_compiled(compiled, data_dict, start_idx)
            add_net_stats(compiled, data_dict, start_idx)
        except:
            failed_file_count += 1
            # find remainder of i, to nearest job number
            s = i % num_experiments_per_file
            # append the job number (corresponding slurm file) to list
            failed_experiment_identifiers.append(i)
            failed_job_identifiers.append((i-s) / num_experiments_per_file)
            
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
        #make a string to report failures
        failures = '\nthe following list shows #\'s of slurm files that had failed experiments:\n' + str(list(set(failed_job_identifiers))) + '\n'

        # Time difference is originally seconds
        finished = (time.time() - start )/ 60
        info = f'\nit took {round(finished,1)} minutes to compile\nor {round(finished / 60,1)} hours'
        file += info
        print(info)
        info = f'\n(#failed files) / (# total number of experiments) is {failed_file_count} / {NEXPERIMENTS}\nor {100 * round(failed_file_count/NEXPERIMENTS, 1)}% failed'
        file += info
        print(info)
        ending = f'\n{filename_prefix} compilation process finished'
        print(ending)
        timing += ending

        failed_exp = '\nthe following list shows #\'s of experiment files that failed:\n' + str(list(set(failed_experiment_identifiers))) + '\n# corresponds to the # in FNAME in experiment() call'

        #only write to the file once, the file will close automatically
        with open(f'{filename_prefix}_compiling_notes.txt','w') as f:
            f.write(file + failures + timing + failed_exp)


def empty_result_dict(num_experiments, nets_per_experiment):
    """ Make empty dictionary for compiling data """
    empty = {}
    nentries = num_experiments * nets_per_experiment
    for col in COLNAMES + NETCOLS:
        empty[col] = [None] * nentries
    empty["exp_num"] = [-1] * nentries
    return empty

def add_to_compiled(compiled, data_dict, start_idx):
    """ Add output dictionary to compiled data, return next empty index """
    for k in data_dict.keys():
        for colname in COLNAMES + NETCOLS:
            compiled[colname][start_idx + k] = data_dict[k][colname]
            
def add_net_stats(compiled, data_dict, start_idx):
    """ Get data from adjacency matrix and add to dict """
    for k in data_dict.keys():
        A = data_dict[k]['adj']
        # Get stats
        g = nx.DiGraph(A.T)
        n = A.shape[0]
        scc = [list(c) for c in nx.strongly_connected_components(g)]
        scc_sz = [len(c) for c in scc]
        wcc = [list(c) for c in nx.weakly_connected_components(g)]
        wcc_sz = [len(c) for c in wcc]
        # Diameter of the largest scc
        diam = nx.diameter(nx.subgraph(g, scc[np.argmax(scc_sz)]))
        # Add to dictionary
        compiled["max_wcc"][start_idx + k] = np.max(wcc_sz)/n
        compiled["max_scc"][start_idx + k] = np.max(wcc_sz)/n
        compiled["singletons"][start_idx + k] = np.sum(np.array(scc_sz) == 1)
        compiled["nscc"][start_idx + k] = len(scc)
        compiled["nwcc"][start_idx + k] = len(wcc)
        compiled["assort"][start_idx + k] = nx.degree_assortativity_coefficient(g)
        compiled["cluster"][start_idx + k] = nx.average_clustering(g)
        compiled["diam"][start_idx + k] = diam

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

compile_output(DIR, filename_prefix, NEXPERIMENTS, NETS_PER_EXPERIMENT)
