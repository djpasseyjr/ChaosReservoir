import networkx as nx
import numpy as np
import pickle
import pandas as pd

DIR = "#TOPOLOGY_DIRECTORY#"
filename_prefix = "#FNAME#"
total_experiment_number = #NUMBER_OF_EXPERIMENTS#

def compile_output(DIR, filename_prefix, total_experiment_number):
    """
    Compile the data from all the various pkl files

    Parameters:
        DIR                     (str): The directory where the individual experiment.py files will be stored
        filename_prefix         (str): prefix to each filename, an error will be thrown if not specified
        total_experiment_number (int): the total number of experiment files that were created
                                       as described by the final parameter_enumaration_number in
                                       the generate_experiments() function of the parameter_experiments.py file
    """
    # we also need the prefix of the files, or can we use os.listdir()
    # path is probably directory plus filename prefix
    path = DIR + filename_prefix + "_"
    #make an initial dataframe that will be added to by all other output files
    # + '1' to path means take first output file
    first_output = dict(pickle.load(open(path + '0','rb')))
    df = pd.DataFrame(first_output)

    for i in range(1,total_experiment_number):
        #concatenante dataframe
        output = dict(pickle.load(open(path + str(i),'rb')))
        df = pd.concat([df,pd.DataFrame(output)],ignore_index=False,sort=False)

    #because the current index is just the network number for a given index
    # store the index as network number
    # the network number will be is the key in the output dictionary originally
    df['network_number'] = df.index
    #reset the Dataframeindex
    #if done appropriately the new index should correspond to the final number in filename
    # corresponding to that specific

    #write final dataframe to pkl file
    df.to_pickle('compiled_output_' + filename_prefix + '.pkl')

compile_output(DIR, filename_prefix, total_experiment_number)
