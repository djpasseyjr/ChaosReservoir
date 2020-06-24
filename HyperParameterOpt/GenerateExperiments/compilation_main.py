
NEXPERIMENTS = #NUMBER_OF_EXPERIMENTS#
PARTITION_NUM = #TODO#

def range_inator(max_experiments,nsplit):
    """ """
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

def write_partitions():
    """write partitioned compile output scripts to leverage
    multiple processors to pcompile data from big batches

    Write partitioned compilation files to compile output in parallel

    Write partitioned files according to the following name with zero
    based indexing
        - 'compiled_output_' + filename_prefix + '_' part_num + '.pkl
    """
    l = range_inator(NEXPERIMENTS,PARTITION_NUM)
    for tuple in l:
        a,b = tuple
        with open('compilation_output_template.py','r') as f:
            tmpl_str = f.read()
        tmpl_str = tmpl_str.replace("#STARTING_EXPERIMENT_NUMBER#",str(a))
        tmpl_str = tmpl_str.replace("#ENDING_EXPERIMENT_NUMBER#",str(b))
        new_name = 'partition_compilation_' + filename +'.py'
        new_f = open(new_name,'w')
        new_f.write(tmpl_str)
        new_f.close()
