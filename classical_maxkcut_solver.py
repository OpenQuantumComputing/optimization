# Classical (i.e., not Quantum) algorithms to solve the Maximum K-Cut problem.

import networkx as nx
import numpy as np
from cylp.cy import CyCbcModel, CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPModel, CyLPArray

def classical_maxkcut_solver(G, num_partitions, num_threads=4):
    # G: NetworkX graph
    # num_partitions: the number partitions or groups in which we should 
    #                 subdivide the nodes (i.e., the value of K)

    N = len(G)
    model = CyLPModel()
    # Decision variables, one for each node
    x = model.addVariable('x', num_partitions * N, isInt=True)
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
    # Adding the selection constraints
    for i in range(N):
        indices = [i + k * N for k in range(num_partitions)]
        model += x[indices].sum() == 1 
    # Adding the cutting constraints
    for i in z_ind:
        for j in z_ind[i]:
            for k in range(num_partitions):
                shift = k * N
                model += z[z_ind[i][j]] + x[i + shift] + x[j + shift] <= 2
                model += z[z_ind[i][j]] + x[i + shift] - x[j + shift] >= 0
                model += z[z_ind[i][j]] - x[i + shift] + x[j + shift] >= 0
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
    sol = [int(i) for i in mip.primalVariableSolution['x']]
    sol_formatted = []
    for i in range(N):
        indices = [i + k * N for k in range(num_partitions)]
        for j in range(num_partitions):
            if (sol[indices[j]] == 1):
                sol_formatted.append(j)
                break
    
    assert(len(sol_formatted) == N)

    return mip.objectiveValue, sol_formatted
