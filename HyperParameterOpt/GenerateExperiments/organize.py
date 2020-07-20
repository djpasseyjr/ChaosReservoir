import subprocess
import os

"""The goal of this file is to be able to organize and move organize
directories that have immediate children files into children directories """

print('if im using os.listdir to organize, it will include directories where i might not want it to ')

#create function to move .pkl files
""" the best thing for the .pkl organizer would to be to organize according to range-inator
    so I  need to know what the total number of experiments is,
    this is why the main.py file is so important

    then once I have the partitions, make the directories for each partition,
    loop through each file putting it into the new directory

    It would be really cool to just pass in name of the main file, and it's location
    have my code read in the main file and figure out how many partitions I had chosen
    as well as how many experiments there will be

    Then for it to run on all the main files and the topology directories

"""

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
    network_options = ['barab1', 'barab2',
                        'erdos', 'random_digraph',
                        'watts3', 'watts5',
                        'watts2','watts4',
                        'geom', 'no_edges',
                        'chain', 'loop',
                        'ident']
    if network not in network_options:
        raise ValueError('{network} not in {network_options}')

    if network == 'barab1' or network == 'barab2':
        DIR = 'Barabasi'
    if network == 'erdos':
        DIR = 'Erdos'
    if network == 'random_digraph':
        DIR = 'RandDigraph'
    if network in ['watts3', 'watts5', 'watts4', 'watts2']:
        DIR = 'Watts'
    if network == 'geom':
        DIR = 'Geometric'
    if network in ['no_edges', 'chain', 'loop', 'ident']:
        DIR = 'AdditionalTopos'
    return DIR

def range_inator(max_experiments,nsplit):
    """Input is number of experiments and number of desired partitions,
    output is a list of tuples which are the range of each partition"""
    partition = max_experiments//nsplit
    output = []
    for i in range(nsplit):
        if i == 0:
            output.append((0,partition))
        elif i == nsplit-1:
            output.append((output[-1][-1],max_experiments))
        else:
            output.append((output[-1][-1],output[-1][-1]+partition))
    #if max_experiments is max_experiments + 1
    # last = output.pop()
    # a,b = last
    # output.append((a,b-1))
    return output

def move_pkl(filename_prefix, num_experiments,num_partitions,loc=None):
    """ Move the .pkl resulting files into directories by partition

    Parameters:
        filename_prefix     (str):
        num_experiments     (int):
        num_partitions      (int):
        loc                 (str): location to work, if None, then assumed working directory

    """
    if not loc:
         loc = ''
    else:
        if loc[-1] != '/':
            loc += '/'

    l = range_inator(num_experiments,num_partitions)

    #make directories
    for i in range(num_partitions):
        subprocess.run(['mkdir',loc + f'{filename_prefix}_result_files_{i}'])

    #move files to directories
    for i,t in enumerate(l):
        a,b = t
        for j in range(a,b):
            subprocess.run(['mv',loc + f'{filename_prefix}_{j}.pkl',loc + f'{filename_prefix}_result_files_{i}/'])

#I could probably run this program with multiple calls to this function and just use the loc parameter


#create function to organize partial datasets
#   don't move the last partial dataset from each partition index, I want to investigate those

#organize the slurm files
