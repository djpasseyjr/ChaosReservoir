import subprocess
import os
import time
import pandas as pd
import datetime as dt
import glob #for using wildcards
import tarfile #may not work in the supercomputer
import numpy as np

def get_subdirectories():
    """ """
    subfolders = [f.name for f in os.scandir() if f.is_dir()]
    if '__pycache__' in subfolders:
        subfolders.remove('__pycache__')
    if '.ipynb_checkpoints' in subfolders:
        subfolders.remove('.ipynb_checkpoints')
    return subfolders

def directory_lengths(write_results=True):
    """ Get a table of subdirectories from working directory and the number of files in each subdirectory

    Parameters:
        write_results (bool)

    Return Table
    """
    subfolders = get_subdirectories()
    results = dict()
    for i,d in enumerate(subfolders):
        results[i] = {'directory':d,'len':len(os.listdir(d))}

    df = pd.DataFrame(results).T
    if write_results:
        with open('dir_lengths_table.txt','w') as f:
            f.write(str(df.sort_values(by='len',ascending=False)))

    return df

def tar_subdirectories(subfolders=None,remove_old=False,verbose=True):
    """
    Parameters:
        subfolders (list): list of directory strings, if None then all subdirectories
        remove_old  (bool): remove old directory after archiving the data
        verbose     (bool): state how long the taring process took
    """
    #tar directories with only data inside it? or just assume
    if subfolders is None:
        subfolders = get_subdirectories()
    results = dict()
    if verbose:
        print(f'there are {len(subfolders)} directories to archive')
    print('do the directories in subfolders need to end in slash?')
    for i,d in enumerate(subfolders):
        dir_length = len(os.listdir(d))
        if  dir_length == 0:
            print(f'{d} has no children')
        else:
            start = time.time()
            mtb = tarfile.open(f'{d}.tar','w')
            mtb.add(d)
            mtb.close()
            min = (time.time() - start) / 60
            if verbose:
                print(f'{d} took {round(min,2)} minutes to tar {dir_length} files')
            if remove_old:
                subprocess.run(['rm','-r',f'{d}'])
                if verbose:
                    print(f'{d} removed')
            results[i] = {'dir':d,'len':dir_length,'min':round(min,2)}
    if verbose:
        df = pd.DataFrame(results).T
        with open('tar_subfolders_results.txt','w') as f:
            f.write(str(df.sort_values(by='min',ascending=False)))

    print('make sure the tar file isnt zipped, the data should be efficiently accessible ')

def partial_data(tar=True,remove_old=True):
    """
    organize partial datasets

    Parameters:
        tar         (bool): tar up the datasets after moving the move important dataset
        remove_old  (bool): remove the unarchived data

    """
    # get a list of all the partial dataset directories
    subfolders = get_subdirectories()
    # find out what the max filename is
    # pull out the partial dataset with the highest index

    tar_list = [] # list to tar up
    for d in subfolders:
        l = os.listdir(d)

        if len(l) > 0 and 'partial' in d:
            splits = [x.split('.') for x in l]
            titles = [int(x[0].split('_')[-1]) for x in splits]

            # raise
            #check to make sure the index number for the partial datasets is the same for all piles
            # to avoid case where partial_compiled_output_w69_chain_0_" & "partial_compiled_output_w69_chain_1_" are in same directory
            # TODO
            # or have it get the max from each of those indices


            file_prefix_list = splits[0][0].split('_')[:-1]
            #recreate the name without the file index
            #ex "partial_compiled_output_w69_chain_0_"
            name_base = ''
            for i in file_prefix_list:
                name_base += i
                name_base += '_'
            nums = np.array(titles)
            max = np.max(nums)
            # move the max file
            if d[-1] == '/':
                subprocess.run(['mv',f'{d + name_base + str(max)}.pkl',f'{name_base + str(max)}.pkl'])
            else:
                subprocess.run(['mv',f'{d}/{name_base + str(max)}.pkl',f'{name_base + str(max)}.pkl'])
            tar.append(d)
        else:
            print(f'{d} has no data or isnt a partial dataset directory')

    # tar up the directory
    # delete the old directory
    if tar:
        # might be faster to know how many files in each directory before taring them up
        directory_lengths(write_results=True)
        # only tar up partial dataset directories
        tar_subdirectories(tar_list,remove_old=remove_old,verbose=True)

def separate(l):
    """Separate slurm files from other files by name, called by mv_slurm function

    Parameters:
        l   (list): list of filenames in a certain directory

    Returns:
        (slurms,others): tuple of lists where
                slurms is a list containing file names of slurm files
                others is a list containing file names of non-slurm files
    """
    slurms = []
    others = []
    for i in l:
        if 'slurm-' in i:
            slurms.append(i[6:].split('_'))
        else:
            others.append(i)

    return (slurms,others)

def slurm_batches(loc):
    """
    Investigate the batch numbers within one directory

    Parameters:
        loc                 (str): location to work, if None, then assumed working directory

    """
    if not loc:
         loc = ''
    else:
        if loc[-1] != '/':
            loc += '/'

    directory_list = os.listdir(loc)

    #parse the slurn names
    files = [a.split('.') for a in directory_list]
    f = [l[0] for l in files]
    file_endings = set([l[1] for l in files])

    l , other = separate(f)

    #move non_slurm files into other directory
    #wildcards in subprocess could be more efficient, not worth investigating though
    if len(other) > 0:
        print('other files are in slurm directory')
        print(other)
        print('\n')

    s = pd.DataFrame(l,columns=['batch_num','file_num'])
    unique_batch_numbers = s['batch_num'].unique()
    print('slurm batch counts \n')
    print(s['batch_num'].value_counts())
    #construct num_to_name dictionary
    print('num_to_names = {')
    for i in s.batch_num:
        print(f'\t{i}:\'\'')
    print('}')
    print('add commas')

def mv_slurm(loc=None,num_to_name=None):
    """
    Assuming a large number of slurm files from different batches are in one directory then organize that directory
    slurm-

    Parameters:
        loc                 (str): location to work, if None, then assumed working directory
        num_to_name         (dict): dictionary containing batch numbers to names for directories,
    """
    start = time.time()
    if not loc:
         loc = ''
         directory_list = os.listdir()
    else:
        if loc[-1] != '/':
            loc += '/'
            directory_list = os.listdir(loc)

    #parse the slurn names
    files = [a.split('.') for a in directory_list]
    f = [l[0] for l in files]
    # there might be a directory therefore not a [1] index
    #list comprehension is faster, but for loop allows for more control
    file_endings = set()
    for l in files:
        try:
            file_endings.add(l[1])
        except:
            pass
    l , other = separate(f)

    #move non_slurm files into other directory
    #wildcards in subprocess could be more efficient, not worth investigating though
    if len(other) > 0:
        print(f'there are {len(other)} other files are in slurm directory')
        month, day = dt.datetime.now().month, dt.datetime.now().day
        hour, minute = dt.datetime.now().hour, dt.datetime.now().minute
        name = f'OTHER_{month}_{day}_at_{hour}_{minute}'
        #make directory
        subprocess.run(['mkdir',loc + name])
        #move other files into directory
        for i in other:
            # the other file could be a directory containing slurm
            if 'OTHER' not in i and 'SLURM' not in i:
                #could be .txt, or .png
                for j in file_endings:
                    #not super efficient but it is effective to go through all file_endings,
                    file_name = i + '.' + j
                    subprocess.run(['mv',loc + file_name,loc + f'{name}/'])

    s = pd.DataFrame(l,columns=['batch_num','file_num'])
    unique_batch_numbers = s['batch_num'].unique()
    sbc = s['batch_num'].value_counts()
    print('slurm batch counts \n')
    print(sbc)

    for i in unique_batch_numbers:

        #make directories
        if num_to_name:
            subprocess.run(['mkdir',loc + f'SLURM_{num_to_name[i]}/'])
        else:
            subprocess.run(['mkdir',loc + f'SLURM_{i}/'])
        #there might only be one file from that batch
        if sbc[i] == 1:
            if num_to_name:
                subprocess.run(['mv',loc + f'slurm-{i}.out',loc + f'SLURM_{num_to_name[i]}/'])
            else:
                subprocess.run(['mv',loc + f'slurm-{i}.out',loc + f'SLURM_{i}/'])
        else:
            vals = s.loc[s.batch_num == i].file_num.values
            for j in vals:
                if num_to_name:
                    subprocess.run(['mv',loc + f'slurm-{i}_{j}.out',loc + f'SLURM_{num_to_name[i]}/'])
                else:
                    subprocess.run(['mv',loc + f'slurm-{i}_{j}.out',loc + f'SLURM_{i}/'])

    print(f'done moving slurm files in {loc}\n after {round((time.time() - start )/ 60,1)} minutes')

    print('do you want to tar up the slurm directories by batch??')

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

def move_pkl(filename_prefix, num_experiments,num_partitions,loc=None,delete_py=True):
    """ Move the .pkl resulting files into directories by partition

    Parameters:
        filename_prefix     (str):
        num_experiments     (int):
        num_partitions      (int):
        loc                 (str): location to work, if None, then assumed working directory
        delete_py           (bool): if True, then the .py files will be deleted
    """
    if not loc:
         loc = ''
    else:
        if loc[-1] != '/':
            loc += '/'

    l = range_inator(num_experiments,num_partitions)

    if delete_py:
        for file in glob.glob(loc + f"{filename_prefix}_*.py"):
            os.remove(file)

    #make directories
    for i in range(num_partitions):
        subprocess.run(['mkdir',loc + f'{filename_prefix}_result_files_{i}'])

    #move files to directories
    for i,t in enumerate(l):
        a,b = t
        for j in range(a,b):
            #doesn't throw an error if file doesn't exist, just has `returncode=1` as output which isn't stored in this case
            subprocess.run(['mv',loc + f'{filename_prefix}_{j}.pkl',loc + f'{filename_prefix}_result_files_{i}/'])

    if loc == '':
        working_directory_name = os.getcwd()
    else:
        working_directory_name = loc
    print(f'done moving {filename_prefix}.pkl files in {working_directory_name}\n')

def update_partition_scripts(filename_prefix,num_partitions,copy_files=False):
    """Update the compile partition based upon the .pkl files being moved

    Parameters:
        filename_prefix     (str):
        num_partitions      (int):
        copy_files          (bool): make duplicates of the partition_compilation files to avoid being overwritten

    """
    if copy_files:
        # print('how to search for just directories using subprocess, as to verify if the copy directory has already been made ')
        # print('should copies include date & time, thats kinda weird cuz I expect this to be run only once except during development')
        #make a directory for the files
        #copy the files using subprocess
        #link: https://stackoverflow.com/questions/7419665/python-move-and-overwrite-files-and-folders?rq=1
        new_directory = f'old_partition_compilation_{filename_prefix}_scripts'
        subprocess.run(['mkdir',new_directory])
        #copy files
        for i in range(num_partitions):
            pc_script = 'partition_compilation_' + filename_prefix + "_" + str(i) + '.py'
            subprocess.run(['cp',pc_script,new_directory + '/' + pc_script])
        print('finished copying pc files')

    #need number of parititons
    for i in range(num_partitions):
        pc_script = 'partition_compilation_' + filename_prefix + "_" + str(i) + '.py'
        topo = filename_prefix.split('_')
        dir = directory(topo[1])

        with open(pc_script,'r') as pcs:
            file_str = pcs.read()


        find = f"DIR = \"{dir}\""
        new = f"DIR = \"{dir}/{filename_prefix}_result_files_{i}\""
        #can't forget to reassign the string or nothing happens
        file_str = file_str.replace(find,new)

        new_name = 'partition_compilation_' + filename_prefix + "_" + str(i) + '.py'
        new_f = open(new_name,'w')
        new_f.write(file_str)
        new_f.close()

    print('finished updating pc scripts')

def batch_pkl_movement(d,verbose=True):
    """ Organize the different topology directories

    Parameters:
        d           (dict): According to format below, as inputs for move_pkl
        verbose     (str)

        #fnp = file_name_prefix
        d = {'file_name_prefix':{
                'num_experiments':60000
                ,'num_partitions':16
                ,'loc':None}
            ,'fnp_2':{
                'num_experiments':70000
                ,'num_partitions':32
                ,'loc':None}
        }
    """
    #an alternative to this function is just to make the call
    # to move_pkl many times from a different script,
    # but this function will time the movements
    # & this function will also do the update_partition_scripts
    # which is convenient
    raise NotImplementedError('not finished')

    for i in d.keys():
        start = time.time()

        move_pkl(filename_prefix=i,
        num_experiments=d[i]['num_experiments'],
        num_partitions=d[i]['num_partitions'],
        loc=d[i]['loc'])

        runtime = time.time() - start
        if verbose:
            print(f'finished with {i} after {round((time.time() - start )/ 60,1)} minutes')

        update_partition_scripts(filename_prefix=i,num_partitions=d[i]['num_partitions'])
        if verbose:
            print(f'finished updating partition scripts for {i}')
