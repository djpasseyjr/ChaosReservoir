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
**Week 1**

__Presentation:__
* [ ] Introduction to Reservoir Computers, Network Topology and Lorentz Equations

__Tasks:__
* [ ] Identify reasonable parameter ranges for the system
* [ ] Send the ranges to a designated group member
* [ ] Designated member generates experiments for all parameter ranges

**Week 2**

__Presentation:__
* [ ] How to Use the Super Computer

__Tasks:__
* [ ] Designated member sends out experiments
* [ ] Each team member runs the experiments and starts data analysis

**Week 3**

__Presentation:__

* [ ] How to Analyze Reservoir Computer Results

__Tasks:__
* [ ] Analyze Datasets

**Week 4**

__Presentation:__
* [ ] Group Members Present Findings

__Tasks:__
* [ ] Further Dataset Analysis

**Week 5**

__Presentation:__
* [ ] Group Members Present Findings

__Tasks:__
* [ ] Discuss Future steps


### Other Directions

1. Heatmap of Wout. What is the correlation between connected components in the original adjacency matrix and size of edges in Wout?
2. Spectral design. Building networks to have certain eigenvalues. Can we start from motifs and connect them in a way that produces the desired structure? Do networks have characteristic motifs that play a role in spectrum as edges are removed.
3. If we have a network and we uniformly remove edges until we break the network down into motifs, can we build it again by adding the edges back in some way?
