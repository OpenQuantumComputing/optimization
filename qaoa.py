from qiskit import execute
import numpy as np
import os
from scipy.optimize import minimize

import sys
sys.path.append('../')

from qiskit_utilities.utilities import *

global g_it
global g_values
global g_bestvalues
global g_gammabeta

### same as getval
#def objective_function(createCircuit, measurementStatistics, params, backend, num_shots, options=None):
#    """
#    :return: minus the expectation value (in order to maximize MaxCut configuration)
#    NB! If a list of circuits are run, only returns the expectation value of the first circuit.
#    """
#
#    qc = createCircuit(params, int(len(params)/2), options=options)
#    if backend.configuration().local:
#        job = execute(qc, backend, shots=num_shots)
#    else:
#        job = start_or_retrieve_job(name, backend, qc, options={'shots' : num_shots})
#    res_data = job.results
#    E,_,_ = measurementStatistics(res_data, options=options)
#    return -E[0]


def getval(gammabeta, createCircuit, measurementStatistics, backend, depth, noisemodel, shots, options):
    global g_it, g_values, g_bestvalues, g_gammabeta
    g_it+=1

    circuit = createCircuit(gammabeta, depth, options=options)

    if backend.configuration().local:
        job = execute(circuit, backend=backend, noise_model=noisemodel, shots=shots)
    else:
        job = start_or_retrieve_job(name+"_"+str(g_it), backend, circuit, options={'shots' : shots})

    val,_,bval = measurementStatistics(job.result().results, options=options)

    g_values[str(g_it)] = val[0]
    g_bestvalues[str(g_it)] = bval
    g_gammabeta[str(g_it)] = gammabeta
    return -val[0]

def runQAOA(createCircuit, measurementStatistics, backend, gamma_n, beta_n, gamma_max, beta_max, optmethod='COBYLA', shots=1024*2*2*2, rerun=True, maxdepth=3, options=None):

    repeats=5
    gammabetas = {}
    E = {}
    best = {}
    name = options.get('name', "None")
### ----------------------------
################
    depth=1
    print("depth =",depth)
################
### ----------------------------
    print("Calculating Energy landscape...")
    gamma_grid = np.linspace(0, gamma_max, gamma_n)
    beta_grid = np.linspace(0, beta_max, beta_n)
    Elandscapefile="../data/sample_graphs/"+name+"_Elandscape.npy"
    if not rerun and os.path.isfile(Elandscapefile):
        Elandscape = np.load(Elandscapefile)
    else:
        if backend.configuration().local:
            circuits=[]
            for beta in beta_grid:
                for gamma in gamma_grid:
                    options['name'] = name+"_"+str(beta_n)+"_"+str(gamma_n)
                    circuits.append(createCircuit(np.array((gamma,beta)), depth, options=options))
            job = execute(circuits, backend, shots=shots)
            El,_,_ = measurementStatistics(job.result().results, options=options)
            Elandscape = -np.array(El)
        else:
            Elandscape = np.zeros((beta_n, gamma_n))
            b=-1
            for beta in beta_grid:
                b+=1
                g=-1
                for gamma in gamma_grid:
                    g+=1
                    options['name'] = name+"_"+str(b)+"_"+str(g)
                    circuit = createCircuit(np.array((gamma,beta)), depth, options = options)
                    job = start_or_retrieve_job(name+"_"+str(b)+"_"+str(g), backend, circuit, options={'shots' : shots})
                    #print("error message = ", job.error_message())
                    #job.error_message()
                    e,_,_ = measurementStatistics(job.result().results, options=options)
                    Elandscape[b,g] = -e[0]
        np.save(Elandscapefile, Elandscape)
    print("Calculating Energy landscape done")

    ### reshape and find parameters that achieved minimal energy
    if backend.configuration().local:
        Elandscape = np.array(Elandscape).reshape(beta_n, gamma_n)
    ind_Emin = np.unravel_index(np.argmin(Elandscape, axis=None), Elandscape.shape)
    x0=np.array((gamma_grid[ind_Emin[1]], beta_grid[ind_Emin[0]]))

    ### local optimization
    #cons = COBYLAConstraints_MaxCut([0,gamma_max], [0,beta_max], depth)
    global g_it, g_values, g_bestvalues, g_gammabeta
    g_it=0
    g_values={}
    g_bestvalues={}
    g_gammabeta={}

    for rep in range(repeats):
        print("depth =",depth, "rep =", rep)
        options['name'] = name+"_opt_"+str(depth)
        out = minimize(getval, x0=x0, method=optmethod, args=(createCircuit, measurementStatistics, backend, depth, None, shots, options), options={'xatol': 1e-2, 'fatol': 1e-1, 'disp': True})#, constraints=cons)
    ### pick the best value along the path
    ind = max(g_values, key=g_values.get)
    gammabetas['x0_d'+str(depth)] = x0.copy()
    gammabetas['xL_d'+str(depth)] = g_gammabeta[ind].copy()
    E[''+str(depth)] = g_values[ind]
    best[''+str(depth)] = g_bestvalues[ind]

    if maxdepth>=2:
### ----------------------------
################
        depth=2
        print("depth =",depth)
################
### ----------------------------

        ### interpolation heuristic
        inter0 = INTERP_init(np.array((gammabetas['xL_d'+str(depth-1)][::2],)))
        inter1 = INTERP_init(np.array((gammabetas['xL_d'+str(depth-1)][1::2],)))
        x0 = np.array((inter0[0], inter1[0], inter0[1], inter1[1]))

        ### local optimization
        #cons = COBYLAConstraints_MaxCut([0,gamma_max], [0,beta_max], depth)
        g_it=0
        g_gammabeta={}
        g_values={}
        g_bestvalues={}

        for rep in range(repeats):
            print("depth =",depth, "rep =", rep)
            options['name'] = name+"_opt_"+str(depth)
            out = minimize(getval, x0=x0, method=optmethod, args=(createCircuit, measurementStatistics, backend, depth, None, shots, options), options={'xatol': 1e-2, 'fatol': 1e-1, 'disp': True})#, constraints=cons)
        ### pick the best value along the path
        ind = max(g_values, key=g_values.get)
        gammabetas['x0_d'+str(depth)] = x0.copy()
        gammabetas['xL_d'+str(depth)] = g_gammabeta[ind].copy()
        E[''+str(depth)] = g_values[ind]
        best[''+str(depth)] = g_bestvalues[ind]

    if maxdepth>=3:
### ----------------------------
################
        depth=3
        print("depth =",depth)
################
### ----------------------------

        ### interpolation heuristic
        inter0 = INTERP_init(gammabetas['xL_d'+str(depth-1)][::2])
        inter1 = INTERP_init(gammabetas['xL_d'+str(depth-1)][1::2])
        x0 = np.array((inter0[0], inter1[0], inter0[1], inter1[1], inter0[2], inter1[2]))

        ### local optimization
        #cons = COBYLAConstraints_MaxCut([0,gamma_max], [0,beta_max], depth)
        g_it=0
        g_gammabeta={}
        g_values={}
        g_bestvalues={}

        for rep in range(repeats):
            print("depth =",depth, "rep =", rep)
            options['name'] = name+"_opt_"+str(depth)
            out = minimize(getval, x0=x0, method=optmethod, args=(createCircuit, measurementStatistics, backend, depth, None, shots, options), options={'xatol': 1e-2, 'fatol': 1e-1, 'disp': True})#, constraints=cons)
        ### pick the best value along the path
        ind = max(g_values, key=g_values.get)
        gammabetas['x0_d'+str(depth)] = x0.copy()
        gammabetas['xL_d'+str(depth)] = g_gammabeta[ind].copy()
        E[''+str(depth)] = g_values[ind]
        best[''+str(depth)] = g_bestvalues[ind]

    if maxdepth>=4:
### ----------------------------
################
        depth=4
        print("depth =",depth)
################
### ----------------------------

        ### interpolation heuristic
        inter0 = INTERP_init(gammabetas['xL_d'+str(depth-1)][::2])
        inter1 = INTERP_init(gammabetas['xL_d'+str(depth-1)][1::2])
        x0 = np.array((inter0[0], inter1[0], inter0[1], inter1[1], inter0[2], inter1[2], inter0[3], inter1[3]))

        ### local optimization
        #cons = COBYLAConstraints_MaxCut([0,gamma_max], [0,beta_max], depth)
        g_it=0
        g_gammabeta={}
        g_values={}
        g_bestvalues={}

        for rep in range(repeats):
            print("depth =",depth, "rep =", rep)
            options['name'] = name+"_opt_"+str(depth)
            out = minimize(getval, x0=x0, method=optmethod, args=(createCircuit, measurementStatistics, backend, depth, None, shots, options), options={'xatol': 1e-2, 'fatol': 1e-1, 'disp': True})#, constraints=cons)
        ### pick the best value along the path
        ind = max(g_values, key=g_values.get)
        gammabetas['x0_d'+str(depth)] = x0.copy()
        gammabetas['xL_d'+str(depth)] = g_gammabeta[ind].copy()
        E[''+str(depth)] = g_values[ind]
        best[''+str(depth)] = g_bestvalues[ind]

    return Elandscape, gammabetas, E, best


def INTERP_init(params_prev_step):
    """
    Takes the optimal parameters at level p as input and returns the optimal inital guess for
    the optimal paramteres at level p+1. Uses the INTERP formula from the paper by Zhou et. al
    :param params_prev_step: optimal parameters at level p
    :return:
    """
    p = params_prev_step.shape[0]
    params_out_list = np.zeros(p+1)
    params_out_list[0] = params_prev_step[0]
    for i in range(2, p + 1):
        # Next line is clunky, but written this way to accommodate the 1-indexing in the paper
        params_out_list[i - 1] = (i - 1) / p * params_prev_step[i-2] + (p - i + 1) / p * params_prev_step[i-1]
    params_out_list[p] = params_prev_step[p-1]
    return params_out_list



def sampleUntilPrecision(circuit,backend,noisemodel,min_n_shots,max_n_shots,E_atol,E_rtol,dv_rtol,confidence_index, measurement_fun, measurement_vars=None):
    """
    Samples from the circuit and calculates the cost function until the specified
    error tolerances are satisfied. This may include several repetitions, either if
    the number of initial shots was too small, or if the variance estimate changed
    to a large degree since the last repetition, meaning that the required shot
    estimate was inaccurate.

    :param circuit: The circuit that will be sampled.
    :param backend: The backend that will execute the circuit.
    :param noisemodel: The noisemodel to use, e.g. when simulating.
    :param min_n_shots: The minimum number of shots to be executed.
    :param max_n_shots: The maximum number of shots to be executed.
    :param E_atol: Absolute error tolerance for the expectation value.
    :param E_rtol: Relative error tolerance for the expectation value.
    :param dv_rtol: Relative change in variance tolerated without repeating.
    :param confidence_index: The degree of confidence required.
    :return: Lists of expectation values, variances and shots each repetition.
    """

    E_tot = 0
    v_tot = 0
    n_tot = 0

    E_list = []
    v_list = []
    n_list = []

    n_req = min_n_shots
    v_prev = v_tot
    while n_tot < n_req and np.abs(v_tot-v_prev) >= dv_rtol*v_prev:
        v_prev = v_tot
        n_cur = n_req - n_tot
        experiment = execute(circuit, backend, noise_model=noisemodel, shots=n_cur)

        [E_cur],[v_cur],_ = measurement_fun(experiment.result().results, options=measurement_vars)
        E_tot = (n_tot*E_tot + n_cur*E_cur)/(n_tot+n_cur)
        v_tot = ((n_tot-1)*v_tot + (n_cur-1)*v_cur)/(n_tot+n_cur-1)
        n_tot = n_req
        E_list.append(E_tot)
        v_list.append(v_tot)
        n_list.append(n_cur)

        E_tol = min(E_atol,E_rtol*E_tot)
        n_req = int(np.ceil(confidence_index**2*v_tot/E_tol**2))

        if n_req > max_n_shots:
            print('Warning: need %d samples to satisfy tolerance %.2e, but max_n_shots = %d.' % (n_req, E_tol, max_n_shots))
            n_req = max_n_shots

    return E_list,v_list,n_list

