import networkx as nx
import numpy as np
import pickle
import pandas as pd

DIR = #TOPOLOGY#
total_experiment_number = #NUMBER_OF_EXPERIMENTS#

def compile_output(path, total_experiment_number):
    """ """
    # we also need the prefix of the files, or can we use os.listdir()
    # path is probably directory plus filename prefix
    path =
    #make an initial dataframe that will be added to by all other output files
    # + '1' to path means take first output file
    first_output = dict(pickle.load(open(path + '1','rb')))
    df = pd.DataFrame(first_output)

    for i in range(total_experiment_number):
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



    raise NotImplementedError('compile_output isnt finished')


compile_output(path, total_experiment_number)
