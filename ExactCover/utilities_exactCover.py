import math
import itertools

def computeOptimalSolution(minus_cost_func, isFeasible_func, FR):
    cost = math.inf
    sol = ""
    nL = FR.shape[1]
    for s in [''.join(i) for i in itertools.product('01', repeat =nL)]:
        if isFeasible_func(s):
            if -minus_cost_func(s)<cost:
                cost = -minus_cost_func(s)
                sol = s

    return cost, sol


def computeAverageApproxRatio(hist, mincost, minus_cost_func):
    #Include the cost of infeasible solutions in average approx ratio or give them approx ratio = 0?
    tot_shots = 0
    avg_approx_ratio = 0
    for key in hist:
        shots = hist[key]
        tot_shots += shots
        cost = -minus_cost_func(key[::-1])   #Qiskit uses big endian encoding, cost function uses litle endian encoding.
                                                    #Therefore the string is reversed before passing it to the cost function.
        approx_ratio = mincost/cost #Only do this to feasible solutions?? And give them 0
        avg_approx_ratio += approx_ratio*shots
    avg_approx_ratio = avg_approx_ratio/tot_shots
    return avg_approx_ratio


    



