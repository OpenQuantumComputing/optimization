# Classical (i.e., not Quantum) algorithms to solve the MaxCut problem.
#
# All functions take as input a NetworkX graph and return the optimal solution.

def enumerate(G):
    if (len(G) > 30):
        raise Exception("Too many solutions to enumerate.")
    
