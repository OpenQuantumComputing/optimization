from openquantumcomputing.mixer_utilities import *
from qiskit import *
from QAOA_class import QAOA
from matplotlib import pyplot as plt
import time, sys, pickle, itertools
import numpy as np
import networkx as nx

T_modes = ["nearest_int", "nearest_int_cyclic", "random", "Hamming", "full"]

for T_mode in T_modes:
    qaoa = QAOA()
    qasm_sim = Aer.get_backend("qasm_simulator")

    # create graph
    graph = nx.read_gml("graph.gml")
    qaoa.graph = graph

    # find feasible states
    feasible_states = np.array([''.join(i) for i in itertools.product('01', repeat = graph.number_of_nodes())])

    # create mixer
    print("create mixer")
    H_mixer = 0
    for i in range(len(feasible_states)):
        for j in range(i + 1, len(feasible_states)):
            T_leftright = get_T(len(feasible_states), "leftright", i = i, j = j)
            H_mixer += Symbol(f"c{i}{j}") * get_Pauli_string_with_algorithm3(feasible_states, T_leftright)

    # find number of cnots
    print("find cnots")
    qaoa.cnots = 0
    for expr in H_mixer:
        qaoa.cnots += num_Cnot(expr)[1]

    # make cost array
    print("make cost array")
    cost_array = np.zeros(len(feasible_states))
    for state in feasible_states:
        cost = 0
        for i, j, data in graph.edges.data():
            if state[int(i)] != state[int(j)]:
                cost += data["weight"]
        cost_array[int(state, 2)] = cost 

    # create mixer array
    print("create mixer array")
    mixer_array = Circuit_maker(H_mixer)

    # parameters
    params = {"graph" : graph, "mixer_array": mixer_array, "cost_array" : cost_array}

    # find mincost
    print("find mincost")
    qaoa.mincost = min(cost_array) 

    # run qaoa
    print("run qaoa")
    depth = 5
    for i in range(depth):
        start = time.time()
        qaoa.increase_depth(backend = qasm_sim, precision = 0.25, params = params)
        print(f"time used on {i} iteration: {time.time() - start}")

    # save data 
    print("savedata")
    with open(f"{T_mode}.pkl", "wb") as output:
        pickle.dump(qaoa, output, pickle.HIGHEST_PROTOCOL)
