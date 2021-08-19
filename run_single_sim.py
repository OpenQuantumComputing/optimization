import numpy as np
import sys
sys.path.append('../')

import os 
from qiskit import *
from qiskit_utilities.utilities import *
from data.tailassignment_loader import *
from tailassignment_oop import *

from tqdm import tqdm

path_to_data = "../data/tailassignment_samples/npy_samples/"

def load_problem(routes, sols, index):
    """
    Loading problem specified by a number of routes,
    number of solutions, and an index.

    Parameters
    ----------
    routes : int
        number of routes
    sols : int
        number of solutions
    index : int 
        index specifying the instance

    Returns
    FR : array
        Constraint matrix
    CR : array
        Cost array

    """

    filename = path_to_data + f"FRCR_{routes}_24_{sols}_{index}.npy"
    FR, CR = npy_loader(filename)
    return FR, CR

def run_statistics_single(QAOA_version,options, simulation_args, file):
    """
    Function for simulating and saving statistics for 
    all the examples in the 'path_to_data' folder with a specified number of 
    routes and solutions and a given index. 
    
    Parameters
    ----------
    QAOA_version : _
        QAOA object
    options : dict
        Options for initializing the tailassignment object 
    simulation_args : dict
        Dictionary with simulation options
    file : string
        File to open.

    Returns
    -------
    SP : array
        Success probability as a function of depth
    C : array
        Energy/cost as a function of depth
    D : int
        Depth of ciruict of one layer
    NCx : int
        Number of Cx-gates in a one-layer circuit
    P : array
        Overlap with the best state as a function of depth

    """
    
    max_depth = simulation_args['max_depth']
    
    FR, CR = npy_loader(path_to_data + file)

    # Normalize weights
    CR /= np.max(CR)

    options['CR'] = CR
    options['FR'] = FR

    qaoa = QAOA_version(options)
    Elandscape, gammabetas, E, best = qaoa.simulate(**simulation_args)
    SP, C, P   = qaoa.simulation_statistics( plot = True, savefig=path_to_data+filename.split(".npy")[0]+".pdf")

    # Normalize the costs
    best_cost = np.max( qaoa.vector_cost(qaoa.state_strings) )
    C       /= best_cost
    
    D, NCx  = qaoa.get_depth_and_numCX()
    
    return SP, C, D, NCx, P

if __name__ == "__main__":

    # Default options and simulation args

    options = dict()
    options['mu'] = 1
    options['usebarrier'] = True

    Aer.backends()
    backend = Aer.get_backend('statevector_simulator')
    
    beta_n    = 50
    gamma_n   = 100

    beta_max  = np.pi
    gamma_max = 2 * np.pi

    optmethod  = 'Nelder-Mead'        
    rerun      = True
    max_depth  = 30

    simulation_args = dict()

    simulation_args['backend']   = backend
    simulation_args['optmethod'] = optmethod
    simulation_args['max_depth'] = max_depth
    simulation_args['repeats']   = 1
    simulation_args['params_ll'] = np.array([0,0])
    simulation_args['params_ul'] = np.array([gamma_max,beta_max])
    simulation_args['params_n']  = np.array([gamma_n, beta_n])

    # Specify the filename and an index when running the program so that it is easily
    # run in parallel, e.g. by the simulate.sh script
    
    filename, index = sys.argv[1], sys.argv[2]

    SP, C, D, NCx, P = run_statistics_single(QAOATailAssignment,options, simulation_args, filename)

    DATA_PATH = "../data/TAstatistics/"
    
    if not os.path.isdir(DATA_PATH):
        os.mkdir(DATA_PATH)

    # Get last part of filename, and use this for saving
    file_sign = filename.split('FRCR')[1]

    np.save(DATA_PATH + f"SP_basic" + file_sign, SP)
    np.save(DATA_PATH + f"C_basic"  + file_sign, C)
    np.save(DATA_PATH + f"P_basic"  + file_sign, P)
    np.save(DATA_PATH + f"D_basic"  + file_sign, D)
    np.save(DATA_PATH + f"NCx_basic"+ file_sign, NCx)

    # New simulation args for the interlaced version
    
    beta_n    = 10
    gamma_n   = 20
    delta_n   = 20

    beta_max  = np.pi
    gamma_max = 2 * np.pi
    delta_max = 2 * np.pi

    optmethod  = 'Nelder-Mead'        
    rerun      = True
    max_depth  = 20

    simulation_args = dict()
    
    simulation_args['backend']   = backend
    simulation_args['optmethod'] = optmethod
    simulation_args['max_depth'] = max_depth
    simulation_args['repeats']   = 1
    simulation_args['params_ll'] = np.array([0,0,0])
    simulation_args['params_ul'] = np.array([gamma_max,beta_max,delta_max])
    simulation_args['params_n']  = np.array([gamma_n, beta_n, delta_n])

    print("Running simulation with interlaced QAOA ... ")
    
    SP, C, D, NCx, P = run_statistics_single(TailAssignmentInterlaced, options, simulation_args, filename)

    np.save(DATA_PATH + f"SP_interlaced"  + file_sign, SP)
    np.save(DATA_PATH + f"P_interlaced"   + file_sign, P)
    np.save(DATA_PATH + f"C_interlaced"   + file_sign, C)
    np.save(DATA_PATH + f"D_interlaced"   + file_sign, D)
    np.save(DATA_PATH + f"NCx_interlaced" + file_sign, NCx)
