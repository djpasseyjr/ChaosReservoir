# THIS README gives instructions for starting on the supercomputer
These instructions assume some basic introduction to Vim (terminal editor), such as being able to navigate using vim, make edits, and save changes by using `esc` + `:wq` 

# setting up environment
after opening the terminal and following the [instructions](https://rc.byu.edu/wiki/?id=Logging+In) to login, then execute the following commands: <br>
`module load python/3.7` <br>
`pip install --user networkx` <br>
`pip install --user sklearn` this library is called in rescomp, it may not even be used, but an error was thrown when this library wasn't downloaded <br>
`module save` this command will make the modules and libraries reside from session to session<br>

# (If not already) download the repos <br>
`cd compute` <br>
`git clone https://github.com/djpasseyjr/ChaosReservoir.git` <br> 
`git clone https://github.com/djpasseyjr/ReservoirSpecialization.git` <br>
see final note about regularly redownloading ChaosReservoir, because it's updated almost daily, as well as being sure to re-locate any valuable data that is wished to be saved before deleting the ChaosReserevoir then recloning from github. <br> If it's been a long time since last downloaded the ChaosReservoir repo, then consider recloning it from github to utilize most recent updates. <br> 

# add rescomp to python path <br>
`cd` will return to home <br>
`vim .bash_profile` (this command will create if it's a non-existent file) open file <br>
`export PYTHONPATH=$PYTHONPATH:/fslhome/jbwilkes/compute/ReservoirSpecialization` 

<font color='red'> note that `jbwilkes` is the user name for joey wilkes, so `jbwilkes` in the export statement should be replaced with the appropriate user id </font> <br>

# to add aliases to quickly switch between popular locations <br>
`cd` will return to home, run the follwing commands from the home directory <br>
`vim .bashrc` <br>
`alias cdbase='cd ~/compute/ChaosReservoir/HyperParameterOpt/GenerateExperiments/'` copy this line into the file, the word "cdbase" can be changed to any command that you consider memorable and convenient, i chose 'base' as referring to homebase, 'cd' + 'base'  because this location is the location where most programs will be executed <br>
`alias cdsd='cd ~/compute/Saved_data'` copy this line into the file, to quickly navigate to location for storing data outside of ChaosReservoir in the case that ChaosReservoir needs to be deleted and recloned with updates, 'cd' + 'sd' where 'sd' abbreviates saved_data <br>
`source ~/.bashrc` make the changes to ".bashrc" active in this session of the terminal, this command needs to be run each session <br>


# PREPARING A BATCH 
`cd ~/compute/ChaosReservoir/HyperParameterOpt/GenerateExperiments/` <br>
observe the "main.py" file, consider looking within the file `vim main.py` <br>

The 'main.py' file is the input for the experiments, it includes the parameter values (in list form) <br>

Make changes to the 'main.py' file according to the experiment you wish to submit, when lots of changes are going to be made, instead of using vim to rewrite the file, I will delete the file entirely , but before doing so I will make a copy of the contents of 'main.py' and paste them into a more convenient text editor (outside the terminal), like atom, then make changes in atom. Then I'll delete the 'main.py' file by running `rm -v main.py`  in the terminal<br> 

Once I've updated the code in atom to reflect the new experiments I wish to run, Then I'll type `vim main.py` which will open a new file, which is empty, i'll copy and paste the code from atom then i'll save the 'main.py' with the changes I wrote. <br>

consider changing the parameter 'WALLTIME_PER_JOB' variable that is in the 'parameter_experiments.py' file (the 10th line), which will dictate how long each experiment has available to run, without having unnecessary wait times in the queue. In the future this variable might be a parameter, that can be adjusted in 'main.py'. 'WALLTIME_PER_JOB' represents the number of minutes for a single job to run. Or this variable can be ignored and the slurm command in the bash script can be changed directly, `#SBATCH --time=`. More information below. <br> 

run the command `python main.py` which will generate the experiment files, as well as the bash script that is needed to run the batch.  <br>

Although everything is ready for the batch to be submitted, one final inspection is recommended (but not required).<br>

Verify the parameters of the batch, by opening the specific bash script. The specific bash script was generated when you ran the 'main.py' with the python command. `ls` will list all the files. Pick the file that ends in `.sh` that has the job_name that was specified in the 'main.py'. <br>

`vim JW7_barab1.sh`, is an example of a specific bash script for running a batch of experiments. <br> 

Pay close attention to the first parameter <br>
`#SBATCH --time=#HOURS#:00:00` where the number of hours should be the max amount of time to run any individual experiments in the batch. <br> where '#HOURS#' was replaced with an integer when 'main.py' was run with python.


# SUBMITTING A BATCH, AKA RUNNING EXPERIMENTS 
after running the main.py file with python, the output will say something similar to 'Next: sbatch JW7_barab1.sh'. That output can be copied and run to submit the job. Use [slurm commands](https://rc.byu.edu/wiki/?id=SLURM+Commands) to see if the job has been run, etc. <br>

Once the job is "running" (according to supercomputer, aka slurm_#####.out files have been created) then the specific cleanup bash script can be run (the exact command is an output from running 'python main.py'). The cleanup file will help organize the files. 

# final note
<font color='blue'> 

UPDATE THE CHAOS_RESERVOIR REPO REGULARLY,  <br>
with the caveat that all data must be removed from  <br>
from the repo before deleting the old version of the repo  <br>
    
</font>

For example, joey created a directory for saving data, so that ChaosReservoir can be rewritten as frequently as necessary, and once the data from a batch has been compiled, then that data is moved to the saved_data directory. 

see Git, to identify when these instructions were last updated
