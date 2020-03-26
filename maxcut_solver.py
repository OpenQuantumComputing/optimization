# Classical (i.e., not Quantum) algorithms to solve the MaxCut problem.
#
# All functions take as input a NetworkX graph and return the optimal solution.

import networkx as nx
import time
import cylp

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

if __name__ == "__main__":
    G = nx.erdos_renyi_graph(15, 0.6)
    for u,v in G.edges():
        G[u][v]['weight'] = 1
    t = time.time()
    print(enumerate(G))
    print(time.time() - t)