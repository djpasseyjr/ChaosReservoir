import pickle
import os
import pandas as pd
from matplotlib import pyplot as plt
import rescomp as rc
import time
#from ChaosReservoir.HyperParameterOpt.GenerateExperiments.res_experiment import * #needed for running locally
from res_experiment import * #used for running in super-computer
import datetime as dt
import pickle
from scipy import integrate

TOL = 5


#small scale is to make sure it runs all the way through
small_scale = False


def lorentz_deriv(t0, X, sigma=10., beta=8./3, rho=28.0):
    """Compute the time-derivative of a Lorenz system."""
    (x, y, z) = X
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

def lorenz_equ(x0=[-20, 10, -.5], begin=0, end=60, timesteps=60000, train_per=.66, clip=0):
    """Use solve_ivp to produce a solution to the lorenz equations"""
    t = np.linspace(begin,end,timesteps)
    clipped_start = floor(timesteps * clip / (end - begin))
    n_train = floor(clipped_start + train_per * (end - clip) / (end - begin) * timesteps)
    train_t = t[clipped_start:n_train]
    test_t = t[n_train:]
    u = integrate.solve_ivp(lorentz_deriv, (begin,end), x0, dense_output=True).sol
    return train_t, test_t, u

def hybrid_metric(pred,true):
    diff = pred - true
    return np.linalg.norm(diff,axis=0) / np.linalg.norm(true,axis=0)

def our_mse(pre,u):
    diff = []
    for i in range(u.shape[1]):
        diff.append(np.sum((u[:,i] - pre[:,i])**2)**.5)
    #mean_error = np.mean(np.linalg.norm(diff, ord=2, axis=0)**2)**(1/2)
    return np.array(diff)

def hybrid_accuracy_duration(hyb_diff):
    """
    Parameters:
        hyb_diff (() ndarray): output of hybrid_metric function
    """
    hybrid_tol = 0.4
    for i in range(len(hyb_diff)):
        dist = hyb_diff[i]
        if dist > hybrid_tol:
            return i
    return len(hyb_diff)

def random_lorenz_x0():
    """ Random initial condition for lorenz equations """
    return  20*(2*np.random.rand(3) - 1)

def metric_comparison_experiments(
    RIDGE_ALPHA = None
    ,SPECT_RAD = None
    ,GAMMA = None
    ,SIGMA  = None
    ,NET = None
    ,TOPO_P = None
    ,REMOVE_P = None
    ,SIZES = [500,1500,2500]
    ,num_distinct_orbits = 20
    ,num_distinct_rescomps = 32
    ,small_scale = False
    ,verbose = False
):
    """ Given a DataFrame of parameters, run experiments
    parameters
        df (DataFrame); dataframe with parameters for experiments as columns
    """
    results = dict()
    assert TOL == 5,'the tolerance should not be changed'

    if small_scale:
        #change adj size to like 1000, and remove_p to 0.9 for all experiments
        df['remove_p'] = 0.9
        df['adj_size'] = 1000

    if None in [RIDGE_ALPHA,SPECT_RAD,GAMMA,SIGMA,NET,REMOVE_P]:
        raise ValueError('Specify Hyper-parameters')

    counter = 0
    # generate 20 distinct orbits

    for size in SIZES:

        DIFF_EQ_PARAMS = {
                  "x0": [-20, 10, -.5],
                  "begin": 0,
                  "end": 155,
                  "timesteps": 155000,
                  "train_per": 100 / 115, #100 seconds of training, so 15 seconds of predict
                  "solver": lorenz_equ,
                  "clip": 40
                 }
        #change the starting position, random orbit
        DIFF_EQ_PARAMS["x0"] = random_lorenz_x0()

        # set the desired parameter combinations

        for _ in range(num_distinct_orbits):
            rc_counter = 0
            for i in range(num_distinct_rescomps):

                RES_PARAMS = {
                              "uniform_weights": True,
                              "solver": "ridge",
                              "ridge_alpha": RIDGE_ALPHA,
                              "signal_dim": 3,
                              "network": "random graph",

                              "res_sz": 15,
                              "activ_f": np.tanh,
                              "connect_p": .4,
                              "spect_rad": SPECT_RAD,
                              "gamma": GAMMA,
                              "sigma": SIGMA,
                              "sparse_res": True,
                             }
                adj = generate_adj(NET,TOPO_P, size)
                # print(z.net, z.topo_p, z.adj_size)

                # Remove Edges
                if REMOVE_P != 0:
                    adj = remove_edges(adj, floor(z.remove_p*np.sum(adj != 0)))

                start = time.time()
                DIFF_EQ_PARAMS["x0"] = random_lorenz_x0()
                train_t, test_t, u = rc_solve_ode(DIFF_EQ_PARAMS)
                rc = ResComp(adj, **RES_PARAMS)
                # Train network
                if small_scale:
                    print('about to start training')
                error = rc.fit(train_t, u)
                if small_scale:
                    print('done training')
                predictions = rc.predict(test_t)
                true = u(test_t)
                pred = how_long_accurate(true, predictions, tol=TOL)
                experiment_time = time.time() - start
                if small_scale:
                    print('minutes to run',experiment_time / 60)

                hyb_diff = hybrid_metric(predictions,true)
                our_diff = our_mse(predictions,true)
                our_accuracy_duration = pred*0.91 / 1000 #scale is in seconds, not time steps so divide by 1k
                their_score_timestep_time = hybrid_accuracy_duration(hyb_diff)
                their_score = their_score_timestep_time * 0.91/1000

                results[counter] = {'prediction' : predictions,
                            'true':true,
                            'pred':pred,
                            'accuracy_duration':our_accuracy_duration,
                            'their_score':their_score,
                            'our_diff':our_diff,
                            'hybrid_diff':hyb_diff,
                            'adj_size': size,
                            'net' : NET,
                            'topo_p' : TOPO_P,
                            'gamma' : GAMMA,
                            'sigma' : SIGMA,
                            'spect_rad' : SPECT_RAD,
                            'ridge_alpha' : RIDGE_ALPHA,
                            'remove_p' : REMOVE_P,
                            'x0':DIFF_EQ_PARAMS['x0'],
                            'rc_counter':rc_counter,
                            'compute time (Min)':experiment_time / 60
                            }
                rc_counter += 1
                if verbose:
                    print('some results')
                    print(results[counter]['accuracy_duration'])
                    print(results[counter]['their_score'])
                    print(results[counter]['compute time (Min)'])
                    print()
                counter += 1
    return results

def save_results(results):
    """Take the output from  metric_comparison_experiments and write it to a file

    """
    #print('would it be better to just write it to a pickle file???')

    try:
        output = pd.DataFrame(results).T
        month, day = dt.datetime.now().month, dt.datetime.now().day
        # hour, minute = dt.datetime.now().hour, dt.datetime.now().minute
        output.to_csv(f'metric_comparison_experiments_{month}_{day}.csv')
    except:
        import sys
        import traceback
        traceback.print_exc()
        print(results)

if __name__ == "__main__":

    results = metric_comparison_experiments(
        RIDGE_ALPHA = 1.e-08
        ,SPECT_RAD = 1
        ,GAMMA = 10
        ,SIGMA  = 0.14
        ,NET = 'erdos'
        ,TOPO_P = 0.5
        ,REMOVE_P = 0.99
        ,SIZES = [500]
        ,num_distinct_orbits = 1
        ,num_distinct_rescomps = 1
        ,verbose = False
    )
    save_results(results)

    message = """ """
    print(message)
