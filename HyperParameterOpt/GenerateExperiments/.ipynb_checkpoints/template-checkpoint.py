from res_experiment import *
from lorenz_sol import *
from math import floor
from scipy import integrate


NTRIALS = 2
NORBITS = 200
X0 = random_lorenz_x0
FNAME = "#FNAME#"
NET = #NET#
REMOVE_P = #REMOVE_P#

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
              "ridge_alpha": .0001,
              "signal_dim": 3,
              "network": "random graph",

              "res_sz": 15,
              "activ_f": np.tanh,
              "connect_p": .4,
              "spect_rad": .9,
              "gamma": 1.,
              "sigma": 0.12,
              "sparse_res": True,
             }

spectrum(FNAME, NET, 
         RES_PARAMS, DIFF_EQ_PARAMS,
         ntrials=NTRIALS,  norbits=NORBITS, 
         x0=X0, remove_p=REMOVE_P)