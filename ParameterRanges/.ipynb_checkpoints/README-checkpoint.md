# Parameter Ranges

Before running the major experiment, we need to identify the ranges of hyper parameters where a 
reservoir computer can reasonably learn the Lorenz equations. We are focusing on the following hyper parameters:

1. From the reservoir computer ODE: `dr/dt = gamma (-r + tanh(Ar + sigma W_in u)`
We are concerned with `gamma`, `sigma`, and the spectral radius of `A`

2. From the ridge regression solver we are concerned with the regularization parameter alpha

3. Network topologies. We are looking at four different network models. Models b and d all accept a connectivity 
parameter that we should tune. Model c accepts a rewiring parameter that we should tune as well.
  
      a. Barabasi-Albert (Preferential attachment)
  
      b. Erdos-Renyi (Undirected Random)
  
      c. Watts-Strogatz (Small World)
   
      d. Random DiGraph 
  
  
