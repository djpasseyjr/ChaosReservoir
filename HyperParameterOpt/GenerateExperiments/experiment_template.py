from res_experiment import *
from rescomp.lorenz_sol import *
from math import floor
from scipy import integrate

#this file is edited automatically by main.py and parameter_experiments.py

DIFF_EQ_PARAMS = {
                  "x0": [-20, 10, -.5],
                  "begin": 0,
                  "end": 60,
                  "timesteps":60000,
                  "train_per": .66,
                  "solver": lorenz_equ
                 }

RES_PARAMS = {
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
    fname="#FNAME#",
    topology="#TOPOLOGY#",
    topo_p=#TOPO_P#,
    res_params=RES_PARAMS,
    diff_eq_params=DIFF_EQ_PARAMS,
    ntrials=#NETS_PER_EXPERIMENT#,
    norbits=#ORBITS_PER_EXPERIMENT#,
    network_size=#SIZE_OF_NETWORK#,
    x0=random_lorenz_x0,
    remove_p=#REMOVE_P#
)
