import networkx as nx
import numpy as np
import pickle
from math import floor, ceil
from rescomp import ResComp, specialize, lorenz_equ
from res_experiment import *
from scipy import sparse

DEBUG = False #similar to verbose but specifically for debugging

def write_dependency_bash(filename_prefix):
    """
    Write a script that will automatically run 3 files with dependencies:
    (1) the data generation batch
    (2) individual_partition_compilation_filenameprefix.sh - the paritioned compilation files
    (3) all_partitions_compilation_filenameprefix.sh - the final compilation

    Parameters:
        filename_prefix (str): essential for running proper files

    Output:
        FILE: `run_' + filename_prefix +'.sh'` which will run the 3 files with dependencies

    """
    with open('bash_dependency_template.sh','r') as f:
        tmpl_str = f.read()
    tmpl_str = tmpl_str.replace("#FILENAME_PREFIX#",filename_prefix)
    # a for auto
    tmpl_str = tmpl_str.replace("#JOB_NAME#",'a_' + filename_prefix)
    new_f = open('run_' + filename_prefix +'.sh','w')
    new_f.write(tmpl_str)
    new_f.close()
    # print('\nbash run_' + filename_prefix +'.sh') #not needed for supermain
    pass

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

def write_bash1(filename,
    number_of_experiments,
    hours_per_job,
    memory_per_job,
):
    """
    Make the bash script 1 that will run the partitioned compile .py files
     """
    with open('bash1_template.sh','r') as f:
        tmpl_str = f.read()
    tmpl_str = tmpl_str.replace("#HOURS#",str(hours_per_job))
    tmpl_str = tmpl_str.replace("#MEMORY#",str(memory_per_job))
    tmpl_str = tmpl_str.replace("#JNAME#",filename[2:] + 'pc')
    tmpl_str = tmpl_str.replace("#FILENAME#",'partition_compilation_' + filename)
    tmpl_str = tmpl_str.replace("#NUMBER_JOBS#",str(number_of_experiments - 1))
    new_f = open('individual_partition_compilation_' + filename +'.sh','w')
    new_f.write(tmpl_str)
    new_f.close()
    # print('written: individual_partition_compilation_' + filename +'.sh')
    pass

def write_bash2(filename,
    hours_per_job,
    memory_per_job,
):
    """
    Write the (optional) bash script to compile all the datasets resulting from
    the individual_partition_compilation*.py files.
     """
    with open('bash2_template.sh','r') as f:
        tmpl_str = f.read()
    tmpl_str = tmpl_str.replace("#HOURS#",str(hours_per_job))
    tmpl_str = tmpl_str.replace("#MEMORY#",str(memory_per_job))
    # JName, as in Job Name.
    tmpl_str = tmpl_str.replace("#JNAME#",filename[2:] + 'bsh2')
    tmpl_str = tmpl_str.replace("#FILENAME#",'merge_partitioned_output_' + filename)
    new_f = open('all_partitions_compilation_' + filename +'.sh','w')
    new_f.write(tmpl_str)
    new_f.close()
    # print('written: all_partitions_compilation_' + filename +'.sh')
    pass

def write_merge(fname,num_partitions):
    """ write the merge file that will compile all the resulting datasets
    from each partition, once the merge file is finished running, then all
    the data for the *filename_prefix* batch has been compiled
    """
    with open('template_merge_compilations.py','r') as f:
        tmpl_str = f.read()
    tmpl_str = tmpl_str.replace("#filename_prefix#",fname)
    tmpl_str = tmpl_str.replace("#partitions#",str(num_partitions))
    new_f = open('merge_partitioned_output_' + fname +'.py','w')
    new_f.write(tmpl_str)
    new_f.close()
    # print('written: merge_partitioned_output_' + fname +'.py')
    pass

def write_partitions(
    PARTITION_NUM,
    compilation_hours_per_partition,
    compilation_memory_per_partition,
    DIR,
    filename_prefix,
    NEXPERIMENTS,
    NETS_PER_EXPERIMENT,
    NUM_EXPERIMENTS_PER_FILE,
    verbose=True,
    bash2_desired=True,
    bash2_walltime_hours = 1,
    bash2_memory_required = 50,
     ):
    """write partitioned compile output scripts to leverage
    multiple processors to pcompile data from big batches

    Write partitioned compilation files to compile output in parallel

    Write partitioned files according to the following name with zero
    based indexing
        - 'compiled_output_' + filename_prefix + '_' part_num + '.pkl
    """
    l = range_inator(NEXPERIMENTS,PARTITION_NUM)
    if DEBUG:
        print(l)
    for i,tuple in enumerate(l):
        a,b = tuple
        with open('compile_dicts_template.py','r') as f:
            tmpl_str = f.read()
        tmpl_str = tmpl_str.replace("#STARTING_EXPERIMENT_NUMBER#",str(a))
        tmpl_str = tmpl_str.replace("#ENDING_EXPERIMENT_NUMBER#",str(b))
        tmpl_str = tmpl_str.replace("#TOPO_DIRECTORY#",DIR)
        tmpl_str = tmpl_str.replace("#FILENAME#",filename_prefix)
        # the number of experiments isn't needed for a partitioned compilation
        tmpl_str = tmpl_str.replace("#NUM_EXPRMTS_PER_FILE#",str(NUM_EXPERIMENTS_PER_FILE))
        tmpl_str = tmpl_str.replace("#NETS_PER_EXPERIMENT#",str(NETS_PER_EXPERIMENT))
        tmpl_str = tmpl_str.replace("#VERBOSE#",str(verbose))
        tmpl_str = tmpl_str.replace("#PARTITION_INDEX#",str(i))
        new_name = 'partition_compilation_' + filename_prefix + "_" + str(i) + '.py'
        new_f = open(new_name,'w')
        new_f.write(tmpl_str)
        new_f.close()
    # print('written all the `partition_compilation_' + filename_prefix + "_*" + '.py` files from 0 to',PARTITION_NUM - 1)

    #write bash_script1
    #filename_prefix
    write_bash1(filename_prefix,
    PARTITION_NUM,
    compilation_hours_per_partition,
    compilation_memory_per_partition)

    if bash2_desired:
        #this might not be desired if the second compilation
        #if the user would prefer to do it locally or in the login node
        write_bash2(filename_prefix,
            bash2_walltime_hours,
            bash2_memory_required)

    write_merge(filename_prefix,PARTITION_NUM)

    # print('\nfinished writing partitions & bash files ')
    pass

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
        raise ValueError(f'{network} not in {network_options}')

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

def write_bash_script(
    directory,
    filename,
    number_of_experiments,
    hours_per_job,
    minutes_per_job,
    memory_per_job,
    ):
    """
    Write the bash script to run all the experiments and write a bash_script to cleanup the directory
    where all the files were created for that batch, write a post_batch completion script to compile output

    Parameters:
        directory               (str): the name of output directory where all resulting pkl files will be stored
        filename                (str): the filename prefix that all the files have in common
        number_of_experiments   (int): the number of experiments is used to systematically
                                        compile all individual output files into one primary file
        hours_per_job               (int): this parameter is passed to write_bash_script
        minutes_per_job             (int): this parameter is passed to write_bash_script
        memory_per_job              (int): Gigabytes, input for --mem-per-cpu slurm command in bash_template for each job in job array
        compile_hours               (int): hours to compile all the output files
        compile_minutes             (int): minutes to compile all the output files
        compile_mem                 (int): Gigabytes, input for --mem-per-cpu slurm command in bash_template

    """
    if minutes_per_job < 0 or minutes_per_job > 60:
        raise ValueError('Minutes per job needs to be between 0-60')
    if not isinstance(memory_per_job,int) and not isinstance(memory_per_job,float):
        raise ValueError('memory should be an int or float')
    if not isinstance(hours_per_job,int) and not isinstance(hours_per_job,float):
        raise ValueError('hours_per_job should be an int or float')

    tmpl_stream = open('bash_template.sh','r')
    tmpl_str = tmpl_stream.read()
    tmpl_stream.close()
    tmpl_str = tmpl_str.replace("#HOURS#",str(hours_per_job))
    tmpl_str = tmpl_str.replace("#MINUTES#",str(minutes_per_job))
    tmpl_str = tmpl_str.replace("#MEMORY#",str(memory_per_job))
    tmpl_str = tmpl_str.replace("#DIR#",directory)
    tmpl_str = tmpl_str.replace("#FNAME#",filename)
    #subtract the number of experiments by one because of zero indexing of filenames
    # whereas the slurm --array range is inclusive on endpoints
    # for example, see https://rc.byu.edu/wiki/index.php?page=How+do+I+submit+a+large+number+of+very+similar+jobs%3F
    #       then search "	Resulting task ID's	" on that webpage to see 0-6 is inclusive on endpoints
    tmpl_str = tmpl_str.replace("#NUMBER_JOBS#",str(number_of_experiments - 1))
    new_f = open(filename +'.sh','w')
    new_f.write(tmpl_str)
    new_f.close()
    # print('NEXT: sg fslg_webb_reservoir \"sbatch',filename +'.sh\"')

    #post_completion script is outdated

    #removed cleanup file

    tmpl_stream = open('template_final_step.sh','r')
    tmpl_str = tmpl_stream.read()
    tmpl_stream.close()
    tmpl_str = tmpl_str.replace("#FNAME#",filename)
    tmpl_str = tmpl_str.replace("#DIR#",directory)
    new_name = 'final_' + filename +'.sh'
    new_f = open(new_name,'w')
    new_f.write(tmpl_str)
    new_f.close()

def generate_experiments(
    FNAME,
    PARTITION_NUM,
    compilation_hours_per_partition,
    compilation_memory_per_partition,
    bash2_desired=True,
    bash2_walltime_hours = 1,
    bash2_memory_required = 50,
    verbose = True,
    nets_per_experiment = 2,
    orbits_per_experiment = 200,
    num_experiments_per_file = 1,
    topology = None,
    hours_per_job = 10,
    minutes_per_job = 0,
    memory_per_job = 3,
    network_sizes = [2000],
    gamma_vals = [1],
    sigma_vals = [0.3],
    spectr_vals = [0.9],
    topo_p_vals = [None],
    ridge_alphas = [0.001],
    remove_p_list = [0.9]
):
    """ Write one bash file (according to bash_template.sh), and individual
    experiment files (according to experiment_template.py) for a grid of
    parameters ranges. See the README for a style guide for fname.

    Parameters:
        FNAME                       (str):   prefix to each filename, an error will be thrown if not specified
        verbose:                    (bool):  print statements to provide extra information
        nets_per_experiment         (int):   number of networks to generate for a given topology
        orbits_per_experiment       (int):   number of orbits to run on each network for a given topology
        num_experiments_per_file    (int):   the number of experiments to write to one .py file, can't be more than the number of permutations of parameters
        topology                    (str):   topology as specified in the generate_adj function of res_experiment.py, an error will be thrown if not specified
        hours_per_job               (int):   this parameter is passed to write_bash_script, including as a parameter provides convenience from main.py
        minutes_per_job             (int):   this parameter is passed to write_bash_script, including as a parameter provides convenience from main.py
        memory_per_job              (int):   Gigabytes, input for --mem-per-cpu slurm command in bash_template
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

    # print('how to make each .py file take about as long as any other .py file to run? just change for loop order?')

    #file count is to index the number of files in an orderly manner
    file_count = 0
    #temp_counter is used to allocate 'num_experiments_per_file' experiments to each .py file
    temp_counter = 0

    a,b,c = len(topo_p_vals), len(gamma_vals), len(sigma_vals)
    d,e,f,g = len(spectr_vals), len(ridge_alphas), len(network_sizes), len(remove_p_list)
    total_experiment_number = a*b*c*d*e*f*g
    if verbose:
        print('\ntotal number of experiments:',total_experiment_number)
    if num_experiments_per_file > total_experiment_number:
        raise ValueError('num_experiments_per_file cant be greater than the total number of possible experiments (product of list length for the 7 parameters)')
    if num_experiments_per_file <= 0:
        raise ValueError('default num_experiments_per_file should be 1, not zero or less')


    parameter_experiment_number = 1
    # print('({parameter_experiment_number},{temp_counter},{file_count})')
    for TOPO_P in topo_p_vals:
        for gamma in gamma_vals:
            for sigma in sigma_vals:
                for spectr in spectr_vals:
                    for ridge_alpha in ridge_alphas:
                        #having network sizes, and remove_p values be the
                        #last 2 values in the loop should mean that each file should have a mix
                        #of low and high network sizes and remove_p values, depending upon 'num_experiments_per_file' of course
                        for n in network_sizes:
                            for p in remove_p_list:
                                #save_fname is for .py files
                                #new_name is for .pkl files
                                save_fname =  DIR + '/' + FNAME + "_" + topology + "_" + str(file_count)
                                new_name = DIR + '/' + FNAME + "_" + topology + "_" + str(parameter_experiment_number)

                                if temp_counter == 0:
                                    # print(f'A:({parameter_experiment_number},{temp_counter},{file_count})')
                                    #read in template experiment file
                                    tmpl_stream = open('job_template.py','r')
                                    tmpl_str = tmpl_stream.read()
                                    tmpl_str = tmpl_str.replace("#FNAME#",new_name + '.pkl')
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
                                    tmpl_stream.close()
                                    #write first file with import statements
                                    new_f = open(save_fname + '.py','w')
                                    new_f.write(tmpl_str)
                                    new_f.close()

                                else:
                                    # print(f'B:({parameter_experiment_number},{temp_counter},{file_count})')
                                    #read in template experiment file
                                    tmpl_stream = open('experiment_template.py','r')
                                    tmpl_str = tmpl_stream.read()
                                    tmpl_str = tmpl_str.replace("#FNAME#",new_name + '.pkl')
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
                                    tmpl_stream.close()
                                    # 'a+' will append to new file
                                    new_f = open(save_fname + '.py','a+')
                                    new_f.write(tmpl_str)
                                    new_f.close()
                                temp_counter += 1
                                parameter_experiment_number += 1

                                if temp_counter >= num_experiments_per_file:
                                    temp_counter = 0
                                    file_count += 1

    # Count the final potentially partial job file
    if temp_counter != 0:
        file_count += 1

    if verbose:
        print('\ntotal number of files/jobs',file_count)
        if DEBUG:
            print('parameter experiment number',parameter_experiment_number)
    #in order to run all the experiments on the supercomputer we need the main bash script
    FNAME_PREFIX = FNAME + "_" + topology
    write_bash_script(DIR,FNAME_PREFIX,file_count,hours_per_job,minutes_per_job,memory_per_job)
    #in order to compile output systematically, store the number of experiments and output directory
    write_partitions(
        PARTITION_NUM,
        compilation_hours_per_partition,
        compilation_memory_per_partition,
        DIR,
        FNAME_PREFIX,
        # subtract one for zero based indexing
        parameter_experiment_number - 1,
        nets_per_experiment,
        num_experiments_per_file,
        verbose,
        bash2_desired,
        bash2_walltime_hours,
        bash2_memory_required,
         )

    write_dependency_bash(FNAME_PREFIX)
    #prepare_output_compilation(DIR,FNAME + "_" + topology,parameter_experiment_number,nets_per_experiment,num_experiments_per_file,verbose)
    pass
