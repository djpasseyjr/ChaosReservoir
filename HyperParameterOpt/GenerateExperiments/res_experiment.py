from specialize import *
from ResComp import ResComp
from matplotlib import pyplot as plt
import sys
import random
import pickle
import warnings
from copy import deepcopy
from utils import timeout, TimeoutError
from scipy import sparse
from networkx import NetworkXError, average_clustering, degree_assortativity_coefficient, diameter
from math import floor
from numpy.linalg import LinAlgError
from lorenz_sol import *


warnings.filterwarnings("ignore", category=FutureWarning)
TOL = 5
MAXTIME = 5*60 # 5 Minutes

def results_dict(keys, res_params, diff_eq_params):
    """ Make dictionaries to store the data
    """
    
    res_data_dict = lambda : {
                                "Fit Error"            : [],
                                "Timesteps Correct"    : [],
                                "Number of Nodes"      : [],
                                "Number of Edges"      : [],
                                "Spec Set"             : [],
                                "Degree"               : [],
                                "Clustring Coeficient" : [],
                                "Degree Assortativity" : [],
                                "Diameter"             : [],
                                "Eigenvalues"          : [],
                                "Centrality"           : [],
                                "Eigenvector Angles"   : []
                             }
    results = {k:res_data_dict() for k in keys}
    results["res_params"] = res_params
    results["diff_eq_params"] = diff_eq_params
    return results
# end

def network_data(A, eigs=False):
    """
    Measure number of nodes, number of edges, egree distribution,
    clustering coeficient, degree assortativity and diameter
    """
    lg      = LightGraph(A)
    size    = lg.n
    edges   = len(lg.edges)
    g       = lg.digraph()
    in_deg  = g.in_degree()
    out_deg = g.out_degree()
    deg_seq = [(in_deg[i],out_deg[i]) for i in range(lg.n)]
    clust_c = average_clustering(g)
    assort  = degree_assortativity_coefficient(g)

    try:
        # Graph needs to be strongly connected
        diam = 0 #diameter(g)
        pass
    except NetworkXError:
        diam = 0

    if eigs:
        if sparse.issparse(A):
            lam, Q = np.linalg.eig(A.toarray())
        else:
            lam, Q = np.linalg.eig(A)
        lam = np.abs(lam)
        cent = Q[:,np.argmax(lam)]
        cent = cent/np.sum(cent)
        angles = np.tril(Q.conj().T.dot(Q))
        angles = angles[np.tril_indices(A.shape[0], k=-1)]
        return size, edges, deg_seq, clust_c, assort, diam, lam, cent, angles
    # end

    return size, edges, deg_seq, clust_c, assort, diam
# end

def rc_store_results(key, results, data,  ctrl=False, eigs=False, base=None):
    """ Add data to the results dictionary
    """
    if ctrl:
        k = key+"_ctrl"
    else:
        k = key

    rc, err, pred = data
    if eigs:
        size, edges, deg_seq, clust_c, assort, diam, lam, cent, angles = network_data(rc.res, eigs=eigs)
    else:
        size, edges, deg_seq, clust_c, assort, diam = network_data(rc.res, eigs=eigs)

    # end
    results[k]["Fit Error"].append(err)
    results[k]["Timesteps Correct"].append(pred)
    results[k]["Number of Nodes"].append(size)
    results[k]["Number of Edges"].append(edges)
    results[k]["Degree"].append(deg_seq)
    results[k]["Clustring Coeficient"].append(clust_c)
    results[k]["Degree Assortativity"].append(assort)
    results[k]["Diameter"].append(diam)
    if eigs:
        results[k]["Eigenvalues"].append(lam)
        results[k]["Centrality"].append(cent)
        results[k]["Eigenvector Angles"].append(angles)
    if base is not None:
        spec = list(set(range(size)) - set(base))
        results[k]["Spec Set"].append(spec)
# end

def rc_save_results(fname, results):
    pickle.dump(results, open(fname,"wb"))
# end

def rc_reset_results(results):
    exp_keys = [k for k in results.keys()]
    exp_keys.remove("res_params")
    exp_keys.remove("diff_eq_params")
    # Find the minimal number of experiments run for any class of experiment
    min_len = min([len(results[exp]["Fit Error"]) for exp in exp_keys])
    # Remove all results beyond the minimal number
    for exp in exp_keys:
        for k,v in results[exp].items():
            results[exp][k] = v[:min_len]
# end

def rc_solve_ode(diff_eq_params):
    """ Wrapper for solving arbitrary ODEs"""
    solver = diff_eq_params.pop("solver")
    sol = solver(**diff_eq_params)
    diff_eq_params["solver"] = solver
    return sol
# end

def how_long_accurate(u, pre, tol=1):
    """ Find the first i such that ||u_i - pre_i||_2 > tol """
    for i in range(u.shape[1]):
        dist = np.sum((u[:,i] - pre[:,i])**2)**.5
        if dist > tol:
            return i
    return u.shape[1]

@timeout(MAXTIME)
def fit_rc(rc, u, train_t, test_t):
    r_0 = rc.state_0
    err = rc.fit(train_t,u)
    pred_len = how_long_accurate(u(test_t), rc.predict(test_t), tol=TOL)
    return rc, r_0, err, pred_len
# end

def initial_rc(u, train_t, test_t, res_params):
    rc = ResComp(**res_params)
    return fit_rc(rc, u, train_t, test_t)
# end

@timeout(MAXTIME)
def rand_spec(rc, u, train_t, test_t, res_params, num_nodes=3, random_W_in=True):
    # Specialize the reservoir
    rc_copy = deepcopy(rc)
    n = rc_copy.res.shape[0]
    base_set = random.sample(list(range(n)),n-num_nodes)
    rc_copy.specialize(base_set, random_W_in=random_W_in)
    return (*fit_rc(rc_copy, u, train_t, test_t), base_set)
# end

@timeout(MAXTIME)
def spec_best(rc, u, train_t, test_t, res_params, num_nodes=3, r_0=None, random_W_in=True):
    scores = rc.score_nodes(train_t, u, r_0=r_0)
    worst_nodes = np.argsort(scores)[:-num_nodes]
    rc.specialize(worst_nodes, random_W_in=random_W_in)
    return (*fit_rc(rc, u, train_t, test_t), worst_nodes)
# end

def rc_control(rc, u, train_t, test_t, res_params):
    param_copy = deepcopy(res_params)
    param_copy["res_sz"] = rc.res.shape[0]
    param_copy["connect_p"] = np.sum(rc.res != 0)/ (rc.res.shape[0]**2)
    rc = ResComp(**param_copy)
    return fit_rc(rc, u, train_t, test_t)
# end

def initial_exper(results, ode_sol, res_params, eigs=False, key="rand"):
    """ Train the initial reservoir computer and store the results of the expriment
        and the statistics of the reservoir.
    """
    train_t, test_t, u = ode_sol
    rc, r0, err, pred = initial_rc(u, train_t, test_t, res_params)
    rc_store_results(key, results, (rc, err, pred), eigs=eigs)
    return rc, r0
# end

def rand_spec_exper(rc, results, ode_sol, res_params, nspec=3, eigs=False, random_W_in=True, key="rand_spec"):
    """ Randomly specializes a reservoir and fits to ode. Stores fit data
        and the statistics of the reservoir. Trains a control random
        reservoir with the same number  of nodes and edges. Stores control
        fit data and statistics.
    """
    train_t, test_t, u = ode_sol
    rc, r0, err, pred, base = rand_spec(rc, u, train_t, test_t, res_params, num_nodes=nspec, random_W_in=random_W_in)
    rc_store_results(key, results, (rc, err, pred), eigs=eigs, base=base)
    rc_ctrl, r0_ctrl, err_ctrl, pred_ctrl = rc_control(rc, u, train_t, test_t, res_params)
    rc_store_results(key, results, (rc_ctrl, err_ctrl, pred_ctrl), ctrl=True, eigs=eigs)
    return rc, r0
# end

def spec_exper(key, rc, r0, results, ode_sol, res_params, nspec=3, eigs=False, random_W_in=True):
    """ Specializes the nspec most useful nodes and fits the new reservoir.
        Stores fit data and statistics of the reservour. Trains a control
        random reservoir with the same number  of nodes and edges. Stores control
        fit data and statistics.
    """
    train_t, test_t, u = ode_sol
    rc, r0, err, pred, base = spec_best(rc, u, train_t, test_t, res_params, num_nodes=nspec, r_0=r0, random_W_in=random_W_in)
    rc_store_results(key, results, (rc, err, pred), eigs=eigs)
    ctrl_rc, ctrl_r0, err, pred = rc_control(rc, u, train_t, test_t, res_params)
    rc_store_results(key,  results, (ctrl_rc, err, pred), ctrl=True, eigs=eigs)
    return rc, r0
# end


def random_lorenz_x0():
    return  20*(2*np.random.rand(3) - 1)
# end

def run_rc_trials(fname,          res_params,
                  diff_eq_params, ntrials=1000,
                  spec_reps=3,    random_W_in=True,
                  nspec=3,        x0=random_lorenz_x0,
                  resume=False,   eigs=False
                 ):

    if resume:
        # Load partially completed dictionary
        results = pickle.load(open(fname,'rb'))
        # Remove partrial results
        rc_reset_results(results)
        # Compute the number of trials remaining
        ntrials -= len(results["rand"]["Fit Error"])
    else:
        # Make dictionary to store data
        keys = ["rand_spec"] + ["spec{}".format(i) for i in range(1,spec_reps+1)]
        keys = keys + [k+"_ctrl" for k in keys]
        keys = keys + ["rand"]
        results = results_dict(keys, res_params, diff_eq_params)

    rep_keys = ["spec{}".format(i) for i in range(1,spec_reps+1)]
    i = 0
    while i < ntrials:
        # Initial condition
        diff_eq_params["x0"] = x0()
        ode_sol = rc_solve_ode(diff_eq_params)

        # Fit reservoir computers
        try:
            rc, r0 = initial_exper(results, ode_sol, res_params, eigs=eigs)
            rand_spec_exper(rc, results, ode_sol, res_params, nspec=nspec, random_W_in=random_W_in, eigs=eigs)
            for key in rep_keys:
                rc, r0 = spec_exper(key, rc, r0, results, ode_sol, res_params, nspec=nspec, random_W_in=random_W_in, eigs=eigs)
            # end

        except TimeoutError:
            print(f"Timeout Error--Network Size: {rc.res.shape[0]}")
            rc_reset_results(results)
            i -= 1
        # end

        i += 1
        rc_save_results(fname, results)
    #end
# end

def rand_leading_order(fname,          res_params,
                  diff_eq_params, ntrials=1000,
                  spec_reps=3,    random_W_in=True,
                  nspec=3,        x0=random_lorenz_x0,
                  resume=False,   eigs=False
                 ):

    if resume:
        # Load partially completed dictionary
        results = pickle.load(open(fname,'rb'))
        # Remove partrial results
        rc_reset_results(results)
        # Compute the number of trials remaining
        ntrials -= len(results["rand"]["Fit Error"])
    else:
        # Make dictionary to store data
        keys = ["rand_spec"] + ["spec{}".format(i) for i in range(1,spec_reps+1)]
        keys = keys + [k+"_ctrl" for k in keys]
        keys = keys + ["rand"]
        results = results_dict(keys, res_params, diff_eq_params)

    rep_keys = ["spec{}".format(i) for i in range(1,spec_reps+1)]
    i = 0
    while i < ntrials:
        # Initial condition
        diff_eq_params["x0"] = x0()
        ode_sol = rc_solve_ode(diff_eq_params)

        # Fit reservoir computers
        try:
            rc, r0 = initial_exper(results, ode_sol, res_params, eigs=eigs)
            for key in rep_keys:
                rc, r0 = rand_spec_exper(rc, results, ode_sol, res_params, nspec=nspec, random_W_in=random_W_in, eigs=eigs, key=key)
            # end

        except TimeoutError:
            print(f"Timeout Error--Network Size: {rc.res.shape[0]}")
            rc_reset_results(results)
            i -= 1
        # end

        i += 1
        rc_save_results(fname, results)
    #end
# end


def rc_trials_percent(fname,          res_params,
                      diff_eq_params, ntrials=1000,
                      spec_reps=3,    random_W_in=True,
                      spec_per=.1,    x0=random_lorenz_x0,
                      resume=False,   eigs=False
                     ):

    if resume:
        # Load partially completed dictionary
        results = pickle.load(open(fname,'rb'))
        # Remove partrial results
        rc_reset_results(results)
        # Compute the number of trials remaining
        ntrials -= len(results["rand"]["Fit Error"])
    else:
        # Make dictionary to store data
        keys = ["rand_spec"] + ["spec{}".format(i) for i in range(1,spec_reps+1)]
        keys = keys + [k+"_ctrl" for k in keys]
        keys = keys + ["rand"]
        results = results_dict(keys, res_params, diff_eq_params)

    rep_keys = ["spec{}".format(i) for i in range(1,spec_reps+1)]
    i = 0
    while i < ntrials:
        # Initial condition
        diff_eq_params["x0"] = x0()
        ode_sol = rc_solve_ode(diff_eq_params)

        # Fit reservoir computers
        try:
            rc, r0 = initial_exper(results, ode_sol, res_params, eigs=eigs)

            nspec = int(floor(spec_per*rc.res.shape[0])) # Compute number of nodes to specialize
            rand_spec_exper(rc, results, ode_sol, res_params, nspec=nspec, random_W_in=random_W_in, eigs=eigs)
            for key in rep_keys:
                nspec = int(floor(spec_per*rc.res.shape[0])) # Compute number of nodes to specialize
                rc, r0 = spec_exper(key, rc, r0, results, ode_sol, res_params, nspec=nspec, random_W_in=random_W_in, eigs=eigs)
            # end

        except TimeoutError:
            print(f"Timeout Error--Network Size: {rc.res.shape[0]}")
            rc_reset_results(results)
            i -= 1
        # end

        i += 1
        rc_save_results(fname, results)
    #end
# end


def compare_topologies(fname,         res_params,
                      diff_eq_params, ntrials=1000,
                      eigs=False,     random_W_in=True,
                      nspec=3,        x0=random_lorenz_x0,
                      resume=False,   
                     ):
    if resume:
        # Load partially completed dictionary
        results = pickle.load(open(fname,'rb'))
        # Remove partrial results
        rc_reset_results(results)
        # Compute the number of trials remaining
        ntrials -= len(results["rand"]["Fit Error"])
    else:
        # Make dictionary to store data
        keys = ["init", "spec", "rand", "pref", "smwd"] 
        results = results_dict(keys, res_params, diff_eq_params)

    i = 0
    n_init = res_params["res_sz"]
    p_init = res_params["connect_p"]
    while i < ntrials:
        # Initial condition
        diff_eq_params["x0"] = x0()
        ode_sol = rc_solve_ode(diff_eq_params)
        

        # Fit reservoir computers
        try:
            # Begin with an initial random graph and specialize it
            res_params["network"]   = "random graph"
            res_params["res_sz"]    = n_init
            res_params["connect_p"] = p_init
            rc, r0 = initial_exper(results, ode_sol, res_params, eigs=eigs, key="init")
            train_t, test_t, u = ode_sol
            rc, r0, err, pred, base = spec_best(rc, u, train_t, test_t, res_params, num_nodes=nspec, r_0=r0, random_W_in=random_W_in)
            rc_store_results("spec", results, (rc, err, pred), eigs=eigs, base=base)
            
            # Compare to other topologies of the same size
            res_params["res_sz"]    = rc.res_sz
            res_params["connect_p"] = np.sum(rc.res != 0)/rc.res_sz**2
            rc, r0 = initial_exper(results, ode_sol, res_params, eigs=eigs, key="rand")
            res_params["network"] = "preferential attachment"
            rc, r0 = initial_exper(results, ode_sol, res_params, eigs=eigs, key="pref")
            res_params["network"] = "small world"
            rc, r0 = initial_exper(results, ode_sol, res_params, eigs=eigs, key="smwd")
            
        except TimeoutError:
            print(f"Timeout Error--Network Size: {rc.res.shape[0]}")
            rc_reset_results(results)
            i -= 1
        # end

        i += 1
        rc_save_results(fname, results)
    #end
# end

def pref_attach():
    rc = ResComp(res_sz=2,   activ_f=np.tanh,
                 connect_p=1.0, ridge_alpha=.00001,
                 spect_rad=.9, sparse_res=True,
                 sigma=0.1,    uniform_weights=True,
                 gamma=1.,     solver="ridge regression",
                 signal_dim=1, network="random graph",
                 max_weight=2, min_weight=0)
    n = np.random.randint(2000,3500)
    return rc.preferential_attachment(n)

def small_world():
    rc = ResComp(res_sz=2,   activ_f=np.tanh,
                     connect_p=1.0, ridge_alpha=.00001,
                     spect_rad=.9, sparse_res=True,
                     sigma=0.1,    uniform_weights=True,
                     gamma=1.,     solver="ridge regression",
                     signal_dim=1, network="random graph",
                     max_weight=2, min_weight=0)   
    n = np.random.randint(2000,3500)
    return rc.small_world(n)

def best_special():
    rc = ResComp(res_sz=15,   activ_f=np.tanh,
                     connect_p=.4, ridge_alpha=.00001,
                     spect_rad=.9, sparse_res=True,
                     sigma=0.1,    uniform_weights=True,
                     gamma=1.,     solver="ridge regression",
                     signal_dim=3, network="random graph",
                     max_weight=2, min_weight=0)
    DIFF_EQ_PARAMS = {
                  "x0": random_lorenz_x0(),
                  "begin": 0,
                  "end": 60,
                  "timesteps":60000,
                  "train_per": .66,
                  "solver": lorenz_equ
                 }
    train_t, test_t, u = rc_solve_ode(DIFF_EQ_PARAMS)
    rc.fit(train_t,u)
    scores = rc.score_nodes(train_t,u)
    base = np.argsort(scores)[:-6]
    rc.specialize(base)
    return rc.res

def aug_spec5():
    sparse.random(30,30, density=.12, dtype=float, format="dok", data_rvs=np.ones)
    for i in range(50):
        n = A.shape[0]
        nodes = list(range(n))
        np.random.shuffle(nodes)
        A, origin = specialize(A,nodes[:-5])
        # Add and remove edges randomly
        nnew_edges = floor(.005*n)
        A = remove_edges(A,nnew_edges)
        A = add_edges(A,nnew_edges)
    return A

def aug_best_spec(per=.2):
    A = best_special()
    if per is None:
        per = .4*np.random.rand()
    nedges = floor(per*len(A))
    return remove_edges(A,nedges)
    
def remove_edges(A,nedges):
    A.todok()
    # Remove Edges
    keys = list(A.keys())
    remove_idx = np.random.choice(range(len(keys)),size=nedges, replace=False)
    remove = [keys[i] for i in remove_idx]
    for e in remove:
        A[e] = 0
    return A
    
def add_edges(A,nedges):
    n = A.shape[0]
    A.todok()
    # Add edges
    x = np.random.choice(range(n), size=nedges)
    y = np.random.choice(range(n), size=nedges)
    for i in range(nedges):
        # Check for self edges
        if x[i] == y[i]:
            y[i] = np.random.randint(x[i],n)
    for e in zip(x,y):
        A[e] = 1.0
    return A

def random_graph():
    n = np.random.randint(2000,3500)
    p = 2/n
    A = nx.adj_matrix(nx.erdos_renyi_graph(n,m)).T
    return sparse.dok_matrix(A)

def spec5():
    A = sparse.rand(30,30, density=.12)
    for i in range(50):
        nodes = list(range(A.shape[0]))
        np.random.shuffle(nodes)
        A, origin = specialize(A,nodes[:-5])
    return A
    
def erdos():
    n = np.random.randint(2000,3500)
    p = 2/n
    return sparse.random(n,n, density=p, data_rvs=np.ones, format='dok')

def barab():
    n = np.random.randint(2000,3500)
    m = 2
    A = nx.adj_matrix(nx.barabasi_albert_graph(n,m)).T
    return sparse.dok_matrix(A)

def watts():
    n = np.random.randint(2000,3500)
    k = 5
    p = .05
    A = nx.adj_matrix(nx.watts_strogatz_graph(n,k,p)).T
    return sparse.dok_matrix(A)

def bspec_rand():
    A = sparse.random(15,15, density=.4, data_rvs=np.ones, format='dok')
    rc = ResComp(A,)
    A, origin = specialize(A,list(range(6)))
    return A

def top3rep():
    rc = ResComp(res_sz=30,   activ_f=np.tanh,
                     connect_p=.12, ridge_alpha=.00001,
                     spect_rad=.9, sparse_res=True,
                     sigma=0.1,    uniform_weights=True,
                     gamma=1.,     solver="ridge regression",
                     signal_dim=3, network="random graph",
                     max_weight=2, min_weight=0)
    DIFF_EQ_PARAMS = {
                  "x0": random_lorenz_x0(),
                  "begin": 0,
                  "end": 60,
                  "timesteps":60000,
                  "train_per": .66,
                  "solver": lorenz_equ
                 }
    train_t, test_t, u = rc_solve_ode(DIFF_EQ_PARAMS)
    while rc.res_sz < 2000:
        rc.fit(train_t,u)
        scores = rc.score_nodes(train_t,u)
        base = np.argsort(scores)[:-3]
        rc.specialize(base)
    return rc.res

def spectrum(fname, network_adj, 
             res_params, diff_eq_params,
             ntrials=1000,  norbits=5, 
             x0=random_lorenz_x0, remove_p=0
            ):    
    
    # Make dictionary to store data
    results = {i:{'net':None, 'pred':[], 'err':[], 'eigs':[]} for i in range(ntrials)}

    i = 0
    while i < ntrials:
        net = network_adj()
        # Remove Edges
        if remove_p != 0:
            net = remove_edges(net,floor(remove_p*np.sum(net != 0)))
        results[i]["net"] = net
        
        # Try to find eigs
        eigs = False
        try:
            results[i]["eigs"] = np.linalg.eigvals(net.toarray())
            eigs = True
            res_params["res_sz"] = net.shape[0]
        except LinAlgError:
            print(f"Eigenvalues didn't converge. Size: {net.shape[0]}")
            i -= 1
            
        # If eigenvalues are found successfully, train the network
        if eigs:
            for j in range(norbits):
                
                # Initial condition
                diff_eq_params["x0"] = x0()
                train_t, test_t, u = rc_solve_ode(diff_eq_params)
                rc = ResComp(net, **res_params)
                
                # Train network
                results[i]["err"].append(rc.fit(train_t,u))
                results[i]["pred"].append(how_long_accurate(u(test_t), rc.predict(test_t), tol=TOL))
        i += 1
        rc_save_results(fname, results)
        print(f"Net complete-- \nNet: {network_adj} \nPercent {remove_p}")

    #end
# end

