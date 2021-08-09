import numpy as np
import sys

sys.path.append('../')
from data.tailassignment_loader import * 

def decompose_problem(FR):

    F, R = np.shape(FR)
    
    routes_per_flight = np.sum(FR, axis = 1)
    fmin              = np.argmin(routes_per_flight)
    min_routes        = int(routes_per_flight[fmin])
    ones_ind          = np.where(FR[fmin,:] == 1)[0]

    compatible_routes = np.zeros((min_routes,R), dtype = bool)
    covered_flights   = np.zeros((min_routes,F), dtype = bool)
    
    for r in range(min_routes):
        current_route      = FR[:,ones_ind[r]]
        covered_flights[r] = FR[:,ones_ind[r]] == 0
        
        for r_ in range(R):
            if FR[:,r_] @ current_route == 0:
                compatible_routes[r,r_] = True

    return ones_ind, compatible_routes, covered_flights
            
        
if __name__ == "__main__":

    instances = 6
    flights   = 24
    solutions = 3

    path_to_examples = "../data/tailassignment_samples/"
    FR, CR, best_sol = load_FR_CR(path_to_examples + f'FRCR_{instances}_{flights}_{solutions}.txt')

    ones_ind, compatible_routes, covered_flights = decompose_problem(FR)

    print(ones_ind)
    for r in range(len(compatible_routes[:,0])):
        print(FR[covered_flights[r,:],compatible_routes[r,:]])
