#merge.py
import numpy as np
import pickle
import pandas as pd
import time
import sys
import traceback

partitions = #partitions#
filename_prefix = "#filename_prefix#"

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

start = time.time()
#Load intial data#
compiled = pickle.load(open('compiled_output_' + filename_prefix + '_0.pkl', 'rb'))

for i in range(1,partitions):
    #Load and merge each compiled partition#
    compiled1 = pickle.load(open('compiled_output_' + filename_prefix + '_' + str(i) + '.pkl', 'rb'))
    compiled = merge_compiled(compiled,compiled1)
    print(f'{i + 1} files combined in {round((time.time() - start )/ 60,1)} minutes')

pickle.dump(compiled, open('completely_compiled_' + filename_prefix + '.pkl', 'wb'))
print(f'{filename_prefix} completely compiled, with export, after {round((time.time() - start )/ 60,1)} minutes')
