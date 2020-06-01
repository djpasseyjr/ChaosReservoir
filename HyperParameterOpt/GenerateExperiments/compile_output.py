import networkx as nx
import numpy as np
import pickle
import pandas as pd
import time

DIR = "#TOPOLOGY_DIRECTORY#"
filename_prefix = "#FNAME#"
total_experiment_number = #NUMBER_OF_EXPERIMENTS#
#verbose will become a parameter in main
verbose = True

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
    path = DIR + "/" + filename_prefix + "_"
    started = False
    failed_file_count = 0
    start = time.time()
    for i in range(total_experiment_number):
        if not started:
            try:
                #make an initial dataframe that will be added to by all other output files
                # + '1' to path means take first output file
                first_output = pickle.load(open(path + str(i)+ '.pkl','rb'))
                # must be transposed for intutive shape, and proper appending
                df = pd.DataFrame(first_output).T
                # the label column is eventually to join with FSL walltime data
                df['label'] = filename_prefix + "_" + str(i)

                #this started will only be reached if the file above exists,
                started = True
            except:
                failed_file_count += 1
        else:
            try:
                #concatenante dataframe
                output = pickle.load(open(path + str(i) + '.pkl','rb'))
                # append doesn't have an "inplace" parameter, so must return copy to df, to make appending reside
                temp = pd.DataFrame(output).T
                temp['label'] = filename_prefix + "_" + str(i)
                df = df.append(temp,ignore_index=False)
            except:
                failed_file_count += 1
            if verbose:
                if i == 0:
                    # don't need to print this
                    pass
                elif i % 1000 == 0:
                    print(f'{i} files compile attempted,\ntime since start (minues):{round((time.time() - start )/ 60,1)}')


    #time difference is originally seconds
    finished = (time.time() - start )/ 60
    #because the current index is just the network number for a given index
    # store the index as network number
    # the network number will be is the key in the output dictionary originally
    df.reset_index(inplace=True)
    df.rename(columns={'index':'network_number'},inplace=True)

    if verbose:
        print(f'it took {round(finished,1)} minutes to compile\nor {round(finished / 60,1)} hours')
        print(f'(#failed files) / (# total number of experiments) is {failed_file_count} / {total_experiment_number}\nor {100 * round(failed_file_count/total_experiment_number,1)}% failed')

    #write final dataframe to pkl file
    df.to_pickle('compiled_output_' + filename_prefix + '.pkl')

    if verbose:
        print(f'{filename_prefix} compilation process finished')

compile_output(DIR, filename_prefix, total_experiment_number)
