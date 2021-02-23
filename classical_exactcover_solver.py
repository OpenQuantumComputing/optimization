# Classical (i.e., not Quantum) algorithms to solve the Exact Cover problem.
#
# The input is a 2-dimensional array A, where aij = 1 if an only if element i is covered by cover j.
# Therefore rows represent elements and columns represent covers.
# The objective is to choose a set of covers such that all elements are covered exactly once.
# An additional input array can be provided to assign a weight to each cover. In this case,
# the exaxct cover with the minium sum of weights will be chosen.

import networkx as nx
import numpy as np
from cylp.cy import CyCbcModel, CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPModel, CyLPArray

def classical_exactcover_solver(A, w=None, num_threads=4):
    nrows, ncolumns = np.shape(A)
    if w is None:
        w = np.ones(ncolumns)
    assert(len(w) == ncolumns)
    assert(sum(w >= 0))
    model = CyLPModel()
    # Decision variables, one for each cover
    x = model.addVariable('x', ncolumns, isInt=True)
    # Adding the box contraints
    model += 0 <= x <= 1
    # Adding the cover constraints
    # Sum_j x_ij ==  1
    for i in range(nrows):
        model += CyLPArray(A[i,:]) * x == 1
    # Adding the objective function
    model.objective = CyLPArray(w) * x
    lp = CyClpSimplex(model)
    lp.logLevel = 0
    lp.optimizationDirection = 'min'
    mip = lp.getCbcModel()
    mip.logLevel = 0
    # Setting number of threads
    mip.numberThreads = num_threads
    mip.solve()

    return mip.objectiveValue, [int(i) for i in mip.primalVariableSolution['x']]
