# Classical (i.e., not Quantum) algorithms to solve the MaxCut problem.
#
# All functions take as input a NetworkX graph and return the optimal solution.

import networkx as nx
import numpy as np
import time
from cylp.cy import CyCbcModel, CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPModel, CyLPArray

def enumerate(G):
    if (len(G) > 30):
        raise Exception("Too many solutions to enumerate.")

    maxcut = []
    maxcut_value = 0
    N = len(G)
    for i in range(2**N - 1):
        x_bin = format(i, 'b').zfill(N)
        x = [int(j) for j in x_bin]
        c = 0
        for u,v in G.edges():
            c += G[u][v]['weight']/2*(1-(2*x[int(u)]-1)*(2*x[int(v)]-1))

        if (c > maxcut_value):
            maxcut = x
            maxcut_value = c

    return maxcut_value, maxcut

def branch_and_bound(G):
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
    mip.numberThreads = 4
    mip.branchAndBound()

    return mip.objectiveValue, [int(i) for i in mip.primalVariableSolution['x']]

if __name__ == "__main__":
    G = nx.erdos_renyi_graph(30, 0.6)
    for u,v in G.edges():
        G[u][v]['weight'] = 1
    # t = time.time()
    # print(enumerate(G))
    # print(time.time() - t)
    t = time.time()
    print(branch_and_bound(G))
    print(time.time() - t)