# ChaosReservoir
Repo for writing and running experiments on the effect of network topology on a reservoir computer's ability to learn the Lorentz equations.

Click [here](Papers/djpassey_thesis.pdf) for the masters thesis that started this project (p. 32-end will be the most relevant). Click [here](Papers/attractor_recon.pdf) for a paper about reservoir computing. Click [here](Papers/spect_dyn_specialization.pdf) for a paper about network specialization.

The code in this repository assumes that the [`rescomp`](https://github.com/djpasseyjr/ReservoirSpecialization) module is installed.

### Group Members
1. Dr. Ben Webb
2. DJ Passey
3. Joseph Jamieson
4. Joey Wilkes

### Plan
* [x] Write code to generate jobs for the super computer
* [x] Write code to compile data from the jobs
* [x] Write code to extract network information from graphs
* [ ] Write statistical analysis pipeline 
* [x] Generate hyper parameter overview jobs
* [x] Run all hyper parameter overview jobs for all topologies
* [ ] Identify the best combination of parameters for each topology
* [ ] Run more extensive tests with the top 1% of parameter combinations
* [ ] Analyze overview data
* [ ] Analyze top preformers data
* [ ] Make plots for paper


### Other Directions

1. What is the correlation between connected components in the original adjacency matrix and size of edges in Wout?
2. Spectral design. Building networks to have certain eigenvalues. Can we start from motifs and connect them in a way that produces the desired structure? Do networks have characteristic motifs that play a role in spectrum as edges are removed.
3. If we have a network and we uniformly remove edges until we break the network down into motifs, can we build it again by adding the edges back in some way?
