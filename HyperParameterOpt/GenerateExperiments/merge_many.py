import numpy as np
import pickle
import pandas as pd
import time
import sys
import traceback

"""this file is for merging many `completely_compiled` datasets from many batches """

#to fill in 
file_list = [

,

]

filename_prefix =

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
compiled = pickle.load(open(file_list[0],'rb'))

for i,f in enumerate(file_list[1:]):
    #Load and merge each compiled partition#
    try:
        compiled1 = pickle.load(open(f, 'rb'))
        compiled = merge_compiled(compiled,compiled1)
        print(f'{i + 1} files combined in {round((time.time() - start )/ 60,1)} minutes')
    except FileNotFoundError:
        print(f'FILE NOT FOUND: {f}')
    except:
        traceback.print_exc()

pickle.dump(compiled, open('best_partial_data_' + filename_prefix + '.pkl', 'wb'))
print(f'{filename_prefix} completely compiled, with export, after {round((time.time() - start )/ 60,1)} minutes')
