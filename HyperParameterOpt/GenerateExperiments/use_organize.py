from organize import *

""" insert function calls from organize.py here """


# move_pkl('jw58_barab1', 33048,4,loc='Barabasi')
# move_pkl('jw59_barab1', 49572,4,loc='Barabasi')
# move_pkl('jw60_watts2', 82620,8,loc='Watts')
# move_pkl('w61_watts2', 82620,8,loc='Watts')
# move_pkl('w62_watts2', 82620,8,loc='Watts')

# update_partition_scripts('jw58_barab1',4,copy_files=False)
update_partition_scripts('jw59_barab1',4,copy_files=False)
update_partition_scripts('jw60_watts2',8,copy_files=True)
update_partition_scripts('w61_watts2',8,copy_files=True)
update_partition_scripts('w62_watts2',8,copy_files=True)

"""
Traceback (most recent call last):
  File "use_organize.py", line 12, in <module>
    update_partition_scripts('jw58_barab1',4)
  File "/lustre/scratch/usr/jbwilkes/ChaosReservoir/HyperParameterOpt/GenerateExperiments/organize.py", line 271, in update_partition_scripts
    with open(pc_script,'rw') as pcs:
ValueError: must have exactly one of create/read/write/append mode
"""
