import matplotlib.pyplot as plt
import networkx as nx
from qaoa import*

def gamma_beta_func_of_p(p, backend, M=5, K = 20, heuristic=False, decimals=0, num_shots=8192):
    fig, (ax1, ax2)= plt.subplots(nrows=2, ncols=1, sharex=True)
    for i in range(M):
        G = nx.random_regular_graph(3, 6)
        addWeights_MaxCut(G, decimals=decimals)
        costs = costsHist_MaxCut(G)
        MAX_COST = max(costs)
        print("Max cost: ", MAX_COST)
        if heuristic:
            params, E = optimize_INTERP(K, G, backend, p, decimals=decimals, num_shots=num_shots)
        else:
            params, E, temp = optimize_random(K, G, backend, p, decimals=decimals, num_shots=num_shots)
        r = E/MAX_COST
        print("Best approximation ratio, r = ", r)
        p_list = np.arange(1, p + 1, 1)
        ax1.scatter(p_list, params[0::2]/np.pi, label=r"$r = %.2f$" %(r))
        ax1.plot(p_list, params[0::2]/np.pi, linestyle="--", alpha=0.4, color="k")
        ax2.scatter(p_list, params[1::2] / np.pi, label=r"$r = %.2f$" % (r))
        ax2.plot(p_list, params[1::2] / np.pi, linestyle="--", alpha=0.4, color="k")

    ax2.set_xlabel(r"$p$")
    ax1.set_ylabel(r"$\gamma/\pi$")
    ax2.set_ylabel(r"$\beta/\pi$")

    ax1.legend()
    ax2.legend()
    plt.show()




def compare_methods(K, G, backend, p_max, decimals=0, num_shots=8192):
    """
    Uses K tries to find the best approx ratio for both INTERP and RI. Does this for all integer p, 1<= p <= p_max
    """
    costs = costsHist_MaxCut(G)
    MAX_COST = max(costs)
    p_list = np.arange(1, p_max + 1, 1)
    E_heur_list = np.zeros(len(p_list))
    E_ran_list = np.zeros(len(p_list))
    for i in range(len(p_list)):
        params_heur, E_heur_list[i] = optimize_INTERP(K, G, backend, p_list[i], decimals=decimals, num_shots=num_shots)
        params_ran, E_ran_list[i], temp = optimize_random(K, G, backend, p_list[i], decimals=decimals, num_shots=num_shots)
    plt.plot(p_list, E_ran_list/MAX_COST, "-o", label="RI")
    plt.plot(p_list, E_heur_list/MAX_COST, "-o", label="INTERP")
    plt.xlabel(r"$p$")
    plt.ylabel(r"$ r^{opt}$")
    plt.legend()
    plt.show()
