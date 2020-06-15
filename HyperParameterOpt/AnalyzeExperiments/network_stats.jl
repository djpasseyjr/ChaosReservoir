import numpy as np
import networkx as nx

COLNAMES = [
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

def empty_result_dict(num_experiments, nets_per_experiment):
    """ Make empty dictionary for compiling data """
    empty = {}
    nentries = num_experiments * nets_per_experiment
    for col in COLNAMES:
        empty[col] = [None] * nentries


def add_to_net_data(net_data, data_dict, start_idx):
    """ Get data from adjacency matrix and add to dict """
    for k in data_dict.keys():
        A = data_dict[k][‘adj’]
        # Get stats
        g = nx.DiGraph(A.T)
        n = A.shape[0]
        scc = [list(c) for c in nx.strongly_connected_components(g)]
        scc_sz = [len(c) for c in scc]
        wcc = [list(c) for c in nx.weakly_connected_components(g)]
        wcc_sz = [len(c) for c in wcc]
        diam =
        # Add to dictionary
        net_data["max_wcc"][start_idx + k] = np.max(wcc_sz)/n
        net_data["max_scc"][start_idx + k] = np.max(wcc_sz)/n
        net_data["singletons"][start_idx + k] = np.sum(np.array(scc_sz) == 1)
        net_data["nscc"][start_idx + k] = len(scc)
        net_data["nwcc"][start_idx + k] = len(wcc)
        net_data["assort"][start_idx + k] = nx.degree_assortativity_coefficient(g)
        net_data["cluster"][start_idx + k] = nx.average_clustering(g)
        net_data["diam"][start_idx + k] =

def net_stats(DIR, filename_prefix, num_experiments, nets_per_experiment):
    """
    Compile the data from all the various pkl files
    Parameters:
        DIR                     (str): The directory where the individual experiment.py files will be stored
        filename_prefix         (str): prefix to each filename, an error will be thrown if not specified
        num_experiments         (int): number of experiments total
        nets_per_experiment     (int): number of nets in each experiment, equivalent to nets_per_experiment in main.py
    """
    # Make dictionary for storing all data
    net_data = empty_result_dict(num_experiments, nets_per_experiment)

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
            add_to_net_data(net_data, data_dict, start_idx)
        except:
            failed_file_count += 1
        # Track experiment number
        for k in range(start_idx, start_idx + nets_per_experiment):
            net_data["exp_num"][k] = i
        start_idx += nets_per_experiment

        if verbose:
            if i % 1000 == 0:
                info = f"\n{i} files compile attempted,time since start (minutes):{round((time.time() - start )/ 60,1)}"
                timing += info
                print(info)
    # write final dict to pkl file
    pickle.dump(compiled, open('compiled_output_' + filename_prefix + '.pkl', 'wb'))
