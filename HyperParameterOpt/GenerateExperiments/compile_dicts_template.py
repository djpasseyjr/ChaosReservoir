#no imports required

PARTITION_NUM = #partition_num#
compilation_hours_per_partition = #compilation_hours_per_partition#
compilation_memory_per_partition = #compilation_memory_per_partition#

#if using a slurm batch to compile partitions then change the following parameters
# first constant is a parameter declaring whether using wants a bash2 script to compile partitions
bash2_desired = False #switch to true if desired
main_compilation_walltime_hours = 1
memory_required = 50


# the following should automatically be filled in
NEXPERIMENTS = #NUMBER_OF_EXPERIMENTS#
NETS_PER_EXPERIMENT = #NUMBER_NETS_PER_EXPERIMENT#
#verbose will become a parameter in main
verbose = #MAIN_VERBOSE#
DIR = "#TOPOLOGY_DIRECTORY#"
filename_prefix = "#FNAME#"

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
    tmpl_str = tmpl_str.replace("#JNAME#",filename[:2] + 'pc')
    tmpl_str = tmpl_str.replace("#FILENAME#",'pc' + filename)
    tmpl_str = tmpl_str.replace("#NUMBER_JOBS#",str(number_of_experiments - 1))
    new_f = open('individual_partition_compilation_' + filename +'.sh','w')
    new_f.write(tmpl_str)
    new_f.close()
    print('written: individual_partition_compilation_' + filename +'.sh')

def write_bash2(filename,
    number_partitions,
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
    tmpl_str = tmpl_str.replace("#JNAME#",filename[:2] + 'bsh2')
    tmpl_str = tmpl_str.replace("#FNAME#",filename[:2] + 'bsh2')
    tmpl_str = tmpl_str.replace("#NUMBER_JOBS#",str(number_of_experiments - 1))
    new_f = open('all_partitions_compilation_' + filename +'.sh','w')
    new_f.write(tmpl_str)
    new_f.close()
    print('written: all_partitions_compilation_' + filename +'.sh')

def write_merge(fname,num_partitions):
    """ write the merge file that will compile all the resulting datasets
    from each partition, once the merge file is finished running, then all
    the data for the *filename_prefix* batch has been compiled
    """
    with open('template_compilation_main.py','r') as f:
        tmpl_str = f.read()
    tmpl_str = tmpl_str.replace("#filename_prefix#",fname)

    tmpl_str = tmpl_str.replace("#partitions#",str(num_partitions))
    new_f = open('merge_partitioned_output_' + fname +'.py','w')
    new_f.write(tmpl_str)
    new_f.close()
    print('written: merge_partitioned_output_' + fname +'.py')

def write_partitions():
    """write partitioned compile output scripts to leverage
    multiple processors to pcompile data from big batches

    Write partitioned compilation files to compile output in parallel

    Write partitioned files according to the following name with zero
    based indexing
        - 'compiled_output_' + filename_prefix + '_' part_num + '.pkl
    """
    l = range_inator(NEXPERIMENTS,PARTITION_NUM)
    for i,tuple in enumerate(l):
        a,b = tuple
        with open('compile_dicts_template.py','r') as f:
            tmpl_str = f.read()
        tmpl_str = tmpl_str.replace("#STARTING_EXPERIMENT_NUMBER#",str(a))
        tmpl_str = tmpl_str.replace("#ENDING_EXPERIMENT_NUMBER#",str(b))
        tmpl_str = tmpl_str.replace("#TOPO_DIRECTORY#",DIR)
        tmpl_str = tmpl_str.replace("#FILENAME#",filename_prefix)
        # the number of experiments isn't needed for a partitioned compilation
        # tmpl_str = tmpl_str.replace("#NUMBER_OF_EXPERIMENTS#",str(number_of_experiments))
        tmpl_str = tmpl_str.replace("#NETS_PER_EXPERIMENT#",str(NETS_PER_EXPERIMENT))
        tmpl_str = tmpl_str.replace("#VERBOSE#",str(verbose))
        tmpl_str = tmpl_str.replace("#PARTITION_INDEX#",str(i))
        new_name = 'partition_compilation_' + filename_prefix + "_" + str(i) + '.py'
        new_f = open(new_name,'w')
        new_f.write(tmpl_str)
        new_f.close()
    print('written all the `partition_compilation_' + filename_prefix + "_*" + '.py` files from 0 to',PARTITION_NUM - 1)

    #write bash_script1
    #filename_prefix
    write_bash1(filename_prefix,
    NEXPERIMENTS,
    compilation_hours_per_partition,
    compilation_memory_per_partition)

    if bash2_desired:
        #this might not be desired if the second compilation
        #if the user would prefer to do it locally or in the login node
        write_bash2(filename_prefix,
            PARTITION_NUM,
            main_compilation_walltime_hours,
            memory_required)

    write_merge(filename_prefix,PARTITION_NUM)

    print('\nfinished writing partitions & bash files ')

write_partitions()
