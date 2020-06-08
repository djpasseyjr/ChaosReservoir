import networkx as nx
import numpy as np
import pickle
from math import floor
from rescomp import ResComp, specialize, lorenz_equ
from scipy import sparse

#-------------------------------------
# Constant for measuring preformance
# Do not change
TOL = 5
#-------------------------------------

smallest_network_size =  int(2e3)
biggest_network_size = int(3.5e3)

#downscale while developing
# smallest_network_size =  int(2)
# biggest_network_size = int(5)
# print(smallest_network_size,biggest_network_size,'was (2000,3500)')

#-- Network topologies --#

def barab1(n=None):
    """ Barabasi-Albert preferential attachment. Each node is added with one edge
    Parameter
        n (int): n is the size of the network
    """
    if n is None:
        n = np.random.randint(smallest_network_size,biggest_network_size)
    m = 1
    A = nx.adj_matrix(nx.barabasi_albert_graph(n,m)).T
    return sparse.dok_matrix(A)

def barab2(n=None):
    """ Barabasi-Albert preferential attachment. Each node is added with two edges
    Parameter
        n (int): n is the size of the network
    """
    if n is None:
        n = np.random.randint(smallest_network_size,biggest_network_size)
    m = 2
    A = nx.adj_matrix(nx.barabasi_albert_graph(n,m)).T
    return sparse.dok_matrix(A)

def erdos(mean_degree,n=None):
    """ Erdos-Renyi random graph.
    Parameter
        mean_degree     (int): specific to this topology
        n               (int): n is the size of the network
    """
    if n is None:
        n = np.random.randint(smallest_network_size,biggest_network_size)
    p = mean_degree/n
    A = nx.adj_matrix(nx.erdos_renyi_graph(n,p)).T
    return sparse.dok_matrix(A)

def random_digraph(mean_degree,n=None):
    """ Random digraph. Each directed edge is present with probability p = mean_degree/n.
        Since this is a directed graph model, mean_degree = mean in deegree = mean out degree

    Parameter
        mean_degree     (int): specific to this topology
        n               (int): n is the size of the network
    """
    if n is None:
        n = np.random.randint(smallest_network_size,biggest_network_size)
    p = mean_degree/n
    return sparse.random(n,n, density=p, data_rvs=np.ones, format='dok')

def watts2(p,n=None):
    """ Watts-Strogatz small world model
    Parameter
        p               (float): specific to this topology
        n               (int): n is the size of the network
    """
    if n is None:
        n = np.random.randint(smallest_network_size,biggest_network_size)
    k = 2
    A = nx.adj_matrix(nx.watts_strogatz_graph(n,k,p)).T
    return sparse.dok_matrix(A)

def watts3(p,n=None):
    """ Watts-Strogatz small world model
    Parameter
        p               (float): specific to this topology
        n               (int): n is the size of the network
    """
    if n is None:
        n = np.random.randint(smallest_network_size,biggest_network_size)
    k = 3
    A = nx.adj_matrix(nx.watts_strogatz_graph(n,k,p)).T
    return sparse.dok_matrix(A)

def watts4(p,n=None):
    """ Watts-Strogatz small world model
    Parameter
        p               (float): specific to this topology
        n               (int): n is the size of the network
    """
    if n is None:
        n = np.random.randint(smallest_network_size,biggest_network_size)
    k = 4
    A = nx.adj_matrix(nx.watts_strogatz_graph(n,k,p)).T
    return sparse.dok_matrix(A)

def watts5(p,n=None):
    """ Watts-Strogatz small world model
    Parameter
        p               (float): specific to this topology
        n               (int): n is the size of the network
    """
    if n is None:
        n = np.random.randint(smallest_network_size,biggest_network_size)
    k = 5
    A = nx.adj_matrix(nx.watts_strogatz_graph(n,k,p)).T
    return sparse.dok_matrix(A)

def geom(mean_degree, n=None):
    """ Random geometric graph
    """
    if n is None:
        n = np.random.randint(smallest_network_size,biggest_network_size)
    r = (mean_degree/(np.pi*n))**.5
    A = nx.adj_matrix(nx.random_geometric_graph(n, r)).T
    return sparse.dok_matrix(A)

def remove_edges(A,nedges):
    """ Randomly removes 'nedges' edges from a sparse matrix 'A'
    """
    A.todok()
    # Remove Edges
    keys = list(A.keys())
    remove_idx = np.random.choice(range(len(keys)),size=nedges, replace=False)
    remove = [keys[i] for i in remove_idx]
    for e in remove:
        A[e] = 0
    return A

def generate_adj(network, param,n=None):
    """ Generate a network with the supplied topology
    Parameters
        network (str)   : one of [barab1, barab2, erdos, random_digraph, watts3, watts5, geom]
        param   (float) : specific to the topology
        n       (int)   : size of the topology, optional

    Returns
        An adjacency matrix with the specified network topology
    """
    # the directory function in parameter_experiments.py needs to have the same
    #       network_options as this function, so if more topologies are added, the directory
    #       function in the other file should also be edited
    network_options = ['barab1', 'barab2',
                        'erdos', 'random_digraph',
                        'watts3', 'watts5',
                        'watts2','watts4',
                        'geom']

    if network not in network_options:
        raise ValueError('{network} not in {network_options}')

    if network == 'barab1':
        return barab1(n)
    if network == 'barab2':
        return barab2(n)
    if network == 'erdos':
        return erdos(param, n)
    if network == 'random_digraph':
        return random_digraph(param, n)
    if network == 'watts3':
        return watts3(param, n)
    if network == 'watts5':
        return watts5(param, n)
    if network == 'watts2':
        return watts2(param, n)
    if network == 'watts4':
        return watts4(param, n)
    if network == 'geom':
        net = geom(param, n)
    return net

#-- Differential equation utilities --#

def random_lorenz_x0():
    """ Random initial condition for lorenz equations """
    return  20*(2*np.random.rand(3) - 1)

def rc_solve_ode(diff_eq_params):
    """ Wrapper for solving arbitrary ODEs"""
    solver = diff_eq_params.pop("solver")
    sol = solver(**diff_eq_params)
    diff_eq_params["solver"] = solver
    return sol

def how_long_accurate(u, pre, tol=1):
    """ Find the first i such that ||u_i - pre_i||_2 > tol """
    for i in range(u.shape[1]):
        dist = np.sum((u[:,i] - pre[:,i])**2)**.5
        if dist > tol:
            return i
    return u.shape[1]

#-- Main experiment --#

def results_dict(*args, **kwargs):
    """ Generate a dictionary for storing experiment results
    """
    ntrials, topology, topo_p, remove_p = args
    results =  {i: {'pred' : [],
                    'err' : [],
                    'mean_pred':None,
                    'mean_err':None,
                    'adj' : None,
                    'adj_size':None,
                    'net' : topology,
                    'topo_p' : topo_p,
                    'gamma' : kwargs['gamma'],
                    'sigma' : kwargs['sigma'],
                    'spect_rad' : kwargs['spect_rad'],
                    'ridge_alpha' : kwargs['ridge_alpha'],
                    'remove_p' : remove_p
                    } for i in range(ntrials)}
    return results

def experiment(
    fname,
    topology,
    topo_p,
    res_params,
    diff_eq_params,
    ntrials=5,
    norbits=5,
    network_size=None,
    x0=random_lorenz_x0,
    remove_p=0
):
    """ Tests the reservoir computers generated by the given hyper parameters
        on 'norbits' different orbits

    Parameters:
        fname (str) : Name of the file where results will be saved
        topology (str) : Network topology in accordance with options in generate_adj()
        topo_p (float) : Parameter accompanying the topology
        res_params (dict) : Dictionary of all parameters for the ResComp class
        diff_eq_params (dict) : Dictionary of all parameters for the rc_solve_ode function
        ntrials (int) : How many different reservoir computers to generate
        norbits (int) : How many orbits per reservoir computer
        network_size (int): Size of the Network Topology
        x0 (function) : Generates an initial condition
        remove_p (float) : Percent of edges to remove from the network
    """
    # Make dictionary to store data
    results = results_dict(ntrials, topology, topo_p, remove_p, **res_params)
    i = 0
    # print('Starting Experiments with the follwing parameters:\n\t', res_params,'\nremove_p',remove_p,)
    while i < ntrials:
        adj = generate_adj(topology, topo_p, network_size)
        results[i]["adj_size"] = adj.shape[0]

        # Remove Edges
        if remove_p != 0:
            adj = remove_edges(adj, floor(remove_p*np.sum(adj != 0)))
        results[i]["adj"] = adj
        # store the size just to see if there is any correlation
        # won't be necessary to store size if we make each topology same size

        for j in range(norbits):
            # Initial condition
            diff_eq_params["x0"] = x0()
            train_t, test_t, u = rc_solve_ode(diff_eq_params)
            rc = ResComp(adj, **res_params)
            # Train network
            results[i]["err"].append(rc.fit(train_t, u))
            results[i]["pred"].append(how_long_accurate(u(test_t), rc.predict(test_t), tol=TOL))

        results[i]['mean_pred'] = np.array(results[i]['pred']).mean()
        results[i]['mean_err'] = np.array(results[i]['err']).mean()
        pickle.dump(results, open(fname,"wb"))
        # print('"Net complete -- \nMean Pred',results[i]['mean_pred'],'\nMean Error',results[i]['mean_err'])
        i += 1
