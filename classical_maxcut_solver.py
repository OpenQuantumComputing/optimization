# Classical (i.e., not Quantum) algorithms to solve the MaxCut problem.
#
# All functions take as input a NetworkX graph and return the optimal solution.

import networkx as nx
import numpy as np
from cylp.cy import CyCbcModel, CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPModel, CyLPArray

def branch_and_bound(G, num_threads=4):
    N = len(G)
    model = CyLPModel()
    # Decision variables, one for each node
    x = model.addVariable('x', N, isInt=True)
    # Adjacency matrix (possibly weighted)
    W = nx.to_numpy_matrix(G)
    z_ind = dict()
    ind = 0
    w = []
    for i in range(N):
        j_range = range(N)
        if (not nx.is_directed(G)):
            # Reduced range for undirected graphs
            j_range = range(i, N)
        for j in j_range:
            if (W[i,j] == 0):
                continue
            if (i not in z_ind):
                z_ind[i] = dict()  
            z_ind[i][j] = ind
            w.append(W[i,j])
            ind += 1
    # Aux variables, one for each edge
    z = model.addVariable('z', len(w), isInt=True)
    # Adding the box contraints
    model += 0 <= x <= 1
    model += 0 <= z <= 1
    # Adding the cutting constraints
    # If x_i == x_j then z_ij = 0
    # If x_i != x_j then z_ij = 1
    for i in z_ind:
        for j in z_ind[i]:
            model += z[z_ind[i][j]] - x[i] - x[j] <= 0
            model += z[z_ind[i][j]] + x[i] + x[j] <= 2
    # Adding the objective function
    model.objective = CyLPArray(w) * z
    lp = CyClpSimplex(model)
    lp.logLevel = 0
    lp.optimizationDirection = 'max'
    mip = lp.getCbcModel()
    mip.logLevel = 0
    # Setting number of threads
    mip.numberThreads = num_threads
    mip.solve()

    return mip.objectiveValue, [int(i) for i in mip.primalVariableSolution['x']]
