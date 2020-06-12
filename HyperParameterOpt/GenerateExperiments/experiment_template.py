# this file purposefully doens't include import statements

DIFF_EQ_PARAMS = {
                  "x0": [-20, 10, -.5],
                  "begin": 0,
                  "end": 85,
                  "timesteps": 85000,
                  "train_per": .889,
                  "solver": lorenz_equ,
                  "clip": 40
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
try:
    #experiment string is for debugging convenience
    experiment = """
    fname="#FNAME#",
    topology="#TOPOLOGY#",
    topo_p=#TOPO_P#,
    ntrials=#NETS_PER_EXPERIMENT#,
    norbits=#ORBITS_PER_EXPERIMENT#,
    network_size=#SIZE_OF_NETWORK#,
    "ridge_alpha": #RIDGE_ALPHA#,
    "spect_rad": #SPECT_RAD#,
    "gamma": #GAMMA#,
    "sigma": #SIGMA#,
    remove_p=#REMOVE_P#
    """
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
except:
    print(experiment)
    traceback.print_exc()
    print('\n\n')
