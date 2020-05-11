from res_experiment import *
from lorenz_sol import *
from math import floor
from scipy import integrate

#this file is edited automatically by main.py and parameter_experiments.py

INPUT_DIFF_EQ_PARAMS = {
                  "x0": [-20, 10, -.5],
                  "begin": 0,
                  "end": 60,
                  "timesteps":60000,
                  "train_per": .66,
                  "solver": lorenz_equ
                 }

INPUT_RES_PARAMS = {
              "uniform_weights": True,
              "solver": "ridge",
              "ridge_alpha": #RIDGE_ALPHA#,
              "signal_dim": 3,
              "network": "random graph",

              "res_sz": 15,
              "activ_f": np.tanh,
              "connect_p": .4,
              "spect_rad": #SPECT_RAD#,
              "gamma": #GAMMA#,
              "sigma": #SIGMA#,
              "sparse_res": True,
             }

experiment(
    FNAME="#FNAME#",
    TOPOLOGY=#TOPOLOGY#,
    TOPO_P=#TOPO_P#,
    RES_PARAMS=INPUT_RES_PARAMS,
    DIFF_EQ_PARAMS=INPUT_DIFF_EQ_PARAMS,
    ntrials=#NETS_PER_EXPERIMENT#,
    norbits=#ORBITS_PER_EXPERIMENT#,
    x0=random_lorenz_x0,
    remove_p=#REMOVE_P#
)
