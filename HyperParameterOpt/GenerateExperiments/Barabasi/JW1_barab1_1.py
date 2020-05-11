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
              "ridge_alpha": 0.001,
              "signal_dim": 3,
              "network": "random graph",

              "res_sz": 15,
              "activ_f": np.tanh,
              "connect_p": .4,
              "spect_rad": 0.9,
              "gamma": 1,
              "sigma": 1,
              "sparse_res": True,
             }

experiment(
    fname="JW1_barab1_1.pkl",
    topology="barab1",
    topo_p=None,
    res_params=RES_PARAMS,
    diff_eq_params=DIFF_EQ_PARAMS,
    ntrials=1,
    norbits=1,
    x0=random_lorenz_x0,
    remove_p=0.1
)
