import pickle
import numpy as np
from matplotlib import pyplot as plt
from openquantumcomputing.QAOABase import QAOABase
from qiskit import *

fig, ax = plt.subplots()

T_modes = ["nearest_int", "nearest_int_cyclic", "random", "hamming", "full"]
for T_mode in T_modes:
    qaoa = pickle.load(open(f"{T_mode}.pkl", "rb"))
    cost_values = np.array(list(qaoa.costval.items()))[:, 1].astype(np.float64)
    alphas = cost_values / qaoa.mincost
    cnots = np.array([qaoa.cnots * (i + 1) for i in range(len(cost_values))]).astype(str)

    x_axis = np.arange(1, len(cost_values) + 1, 1)
    plt.plot(x_axis, alphas, ".-", label = T_mode)
    [ax.annotate(cnot, (x, alpha), verticalalignment="top") for cnot, x, alpha in zip(cnots, x_axis, alphas)]

plt.xticks(x_axis)
plt.xlabel("depth")
plt.ylim(0.4, 1.0)
plt.ylabel(r"$\alpha$")
plt.legend()
plt.show()
