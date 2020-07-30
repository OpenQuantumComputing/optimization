from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
import numpy as np
import networkx as nx
from scipy.optimize import minimize



def createCircuit_MaxCut(x,G,depth,version=1, applyX=[], usebarrier=False):
    num_V = G.number_of_nodes()
    q = QuantumRegister(num_V)
    c = ClassicalRegister(num_V)
    circ = QuantumCircuit(q,c)
    if len(applyX)==0:
        circ.h(range(num_V))
    else:
        if np.where(np.array(applyX)==1)[0].size>0:
            circ.x(np.where(np.array(applyX)==1)[0])
        circ.h(range(num_V))
    if usebarrier:
        circ.barrier()
    for d in range(depth):
        gamma=x[2*d]
        beta=x[2*d+1]
        for edge in G.edges():
            i=int(edge[0])
            j=int(edge[1])
            w = G[edge[0]][edge[1]]['weight']
            wg = w*gamma
            if version==1:
                circ.cx(q[i],q[j])
                circ.rz(wg,q[j])
                circ.cx(q[i],q[j])
            else:
                circ.cu1(-2*wg, i, j)
                circ.u1(wg, i)
                circ.u1(wg, j)
        if usebarrier:
            circ.barrier()
        circ.rx(2*beta,range(num_V))
        if usebarrier:
            circ.barrier()
    circ.measure(q,c)
    return circ

def cost_MaxCut(x,G):
    C=0
    for edge in G.edges():
        i = int(edge[0])
        j = int(edge[1])
        w = G[edge[0]][edge[1]]['weight']
        C = C + w/2*(1-(2*x[i]-1)*(2*x[j]-1))
    return C

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

def listSortedCosts_MaxCut(G):
    costs={}
    maximum=0
    solutions=[]
    num_V = G.number_of_nodes()
    for i in range(2**num_V):
        binstring="{0:b}".format(i).zfill(num_V)
        y=[int(i) for i in binstring]
        costs[binstring]=cost_MaxCut(y,G)
    sortedcosts={k: v for k, v in sorted(costs.items(), key=lambda item: item[1])}
    return sortedcosts

def costsHist_MaxCut(G):
    num_V = G.number_of_nodes()
    costs=np.ones(2**num_V)
    for i in range(2**num_V):
        if i%1024*2*2*2==0:
            print(i/2**num_V*100, "%", end='\r')
        binstring="{0:b}".format(i).zfill(num_V)
        y=[int(i) for i in binstring]
        costs[i]=cost_MaxCut(y,G)
    print("100%")
    return costs

def bins_comp_basis(data, G):
    max_solutions=[]
    num_V = G.number_of_nodes()
    bins_states = np.zeros(2**num_V)
    num_shots=0
    num_solutions=0
    max_cost=0
    average_cost=0
    for item, binary_rep in enumerate(data):
        integer_rep=int(str(binary_rep), 2)
        counts=data[str(binary_rep)]
        bins_states[integer_rep] += counts
        num_shots+=counts
        num_solutions+=1
        y=[int(i) for i in str(binary_rep)]
        lc = cost_MaxCut(y,G)
        if lc==max_cost:
            max_solutions.append(y)
        elif lc>max_cost:
            max_solutions=[]
            max_solutions.append(y)
            max_cost=lc
        average_cost+=lc*counts
    return bins_states, max_cost, average_cost/num_shots, max_solutions


def objective_function(params, G, backend, num_shots=8192):
    """
    :return: minus the expectation value (in order to maximize MaxCut configuration)
    NB! If a list of circuits are ran, only returns the expectation value of the first circuit.
    """
    qc = createCircuit_MaxCut(params, G, int(len(params)/2))
    res_data = execute(qc, backend, shots=num_shots).result().results
    E,_ = measurementStatistics_MaxCut(res_data, G)
    return -E[0]

def random_init(gamma_bounds,beta_bounds,depth):
    """
    Enforces the bounds of gamma and beta based on the graph type.
    :param gamma_bounds: Parameter bound tuple (min,max) for gamma
    :param beta_bounds: Parameter bound tuple (min,max) for beta
    :return: np.array on the form (gamma_1, beta_1, gamma_2, ...., gamma_d, beta_d)
    """
    gamma_list = np.random.uniform(gamma_bounds[0],gamma_bounds[1], size=depth)
    beta_list = np.random.uniform(beta_bounds[0],beta_bounds[1], size=depth)
    initial = np.empty((gamma_list.size + beta_list.size,), dtype=gamma_list.dtype)
    initial[0::2] = gamma_list
    initial[1::2] = beta_list
    return initial

def parameterBounds_MaxCut(G,decimals=0,weight_rtol=1e-3):
    """
    :param G: The weighted or unweighted graph to perform MaxCut on.p
    :param decimals: The number of decimals to keep in the weights.
    :param weight_rtol: The relative error allowed when rounding the weights.
    :return: Bounds of the first periodic domain for gamma and beta.
    """
    scaling_factor = np.power(10,decimals)

    scaled_weights = []
    for _,_,w in G.edges.data('weight',default=1):
        scaled_w = w*scaling_factor
        scaled_w_int = int(round(scaled_w))
        if abs(scaled_w_int-scaled_w) > weight_rtol*scaled_w:
            print('Warning: When finding parameter bounds, rounding the weight %.2e '
                  'to %d decimals, we introduced an error larger than the relative '
                  'tolerance %.2e.' % (w, decimals,weight_rtol))
        scaled_weights.append(scaled_w_int)

    gcd = np.gcd.reduce(scaled_weights)

    gamma_period = 2*np.pi*scaling_factor/gcd
    beta_period = np.pi/2

    gamma_min = 0
    gamma_max = gamma_period/2
    beta_min = 0
    beta_max = beta_period

    return (gamma_min,gamma_max),(beta_min,beta_max)


def wrapParameters_MaxCut(gamma,beta,gamma_bounds,beta_bounds):
    gamma_period = 2*(gamma_bounds[1]-gamma_bounds[0])
    beta_period = beta_bounds[1]-beta_bounds[0]

    gamma = np.mod(gamma,gamma_period)
    beta = np.mod(beta,beta_period)

    if gamma > gamma_period/2:
        gamma = gamma_period - gamma
        beta = beta_period - beta
    return gamma,beta

def spatialFrequencies_MaxCut(G):
    """
    Get the maximum typical frequencies for parameter space
    :param G: The graph with weights.
    :return: tuple with gamma and beta frequencies
    """
    weights = [w for _,_,w in G.edges.data('weight',default=1)]
    gamma_freq = np.linalg.norm(weights,2)/(2*np.pi)
    beta_freq = np.sqrt(G.number_of_nodes())/(np.pi)

    return gamma_freq,beta_freq


def COBYLAConstraints_MaxCut(gamma_bounds,beta_bounds,depth):
    """
    Get constraint list to use with COBYLA.
    :param gamma_bounds: Parameter bound tuple (min,max) for gamma
    :param beta_bounds: Parameter bound tuple (min,max) for beta
    :param depth: Depth of the circuit
    :return: List of constraints applying to the parameters
    """
    constraints = []
    for j in range(depth):
        if j % 2 == 0:
            (lower,upper) = gamma_bounds
        else:
            (lower, upper) = beta_bounds

        lower_constraint = {'type': 'ineq', 'fun': lambda x, lb=lower, i=j: x[i] - lb}
        upper_constraint = {'type': 'ineq', 'fun': lambda x, ub=upper, i=j: ub - x[i]}
        constraints.append(lower_constraint)
        constraints.append(upper_constraint)
    return constraints

def optimize_random(K, G, backend, depth=1, decimals=0, num_shots=8192):
    """
    :param K: # Random initializations (RIs)
    :return: Array of best params (on the format where the gammas and betas are intertwined),
    the corresponding best energy value, and the average energy value for all the RIs
    """
    record = -np.inf
    avg_list = np.zeros(K)
    for i in range(K):
        gamma_bounds, beta_bounds = parameterBounds_MaxCut(G, decimals=decimals)
        init_params = random_init(gamma_bounds, beta_bounds, depth)
        cons = COBYLAConstraints_MaxCut(gamma_bounds, beta_bounds, depth)
        sol = minimize(objective_function, x0=init_params, method='COBYLA', args=(G, backend, num_shots), constraints=cons)
        params = sol.x
        qc = createCircuit_MaxCut(params, G, depth)
        temp_res_data = execute(qc, backend, shots=num_shots).result().results
        [E],_ = measurementStatistics_MaxCut(temp_res_data, G)
        avg_list[i] = E
        if E>record:
            record = E
            record_params = params
    return record_params, record, np.average(avg_list)

def scale_p(K, G, backend, depth=3, decimals=0, num_shots=8192):
    """
    :return: arrays of the p_values used, the corresponding array for the energy from the optimal
         energy config, and the average energy (for all the RIs at each p value)
    """
    H_list = np.zeros(depth)
    avg_list = np.zeros(depth)
    p_list = np.arange(1, depth + 1, 1)
    for d in range(1, depth + 1):
        temp, H_list[d-1], avg_list[d-1] = optimize_random(K, G, backend, d, decimals=decimals, num_shots=num_shots)
    return p_list, H_list, avg_list



def INTERP_init(params_prev_step):
    """
    Takes the optimal parameters at level p as input and returns the optimal inital guess for
    the optimal paramteres at level p+1. Uses the INTERP formula from the paper by Zhou et. al
    :param params_prev_step: optimal parameters at level p
    :return:
    """
    p = len(params_prev_step)
    params_out_list = np.zeros(p+1)
    params_out_list[0] = params_prev_step[0]
    for i in range(2, p + 1):
        # Next line is clunky, but written this way to accommodate the 1-indexing in the paper
        params_out_list[i - 1] = (i - 1) / p * params_prev_step[i-2] + (p - i + 1) / p * params_prev_step[i-1]
    params_out_list[p] = params_prev_step[p-1]
    return params_out_list

def optimize_INTERP(K, G, backend, depth, decimals=0, num_shots=8192):
    """
    Optimizes the params using the INTERP heuristic
    :return: Array of the optimal parameters, and the correponding energy value
    """
    record = -np.inf
    for i in range(K):
        init_params = np.zeros(2)
        gamma_bounds, beta_bounds = parameterBounds_MaxCut(G, decimals=decimals)
        cons = COBYLAConstraints_MaxCut(gamma_bounds, beta_bounds, 1)
        sol = minimize(objective_function, x0=init_params, method='COBYLA', args=(G, backend, num_shots), constraints=cons)
        params = sol.x
        init_gamma = params[0:1]
        init_beta = params[1:2]
        for p in range(2, depth + 1):
            init_gamma = INTERP_init(init_gamma)
            init_beta = INTERP_init(init_beta)
            init_params = np.zeros(2 * p)
            init_params[0::2] = init_gamma
            init_params[1::2] = init_beta
            cons = COBYLAConstraints_MaxCut(gamma_bounds, beta_bounds, p)
            sol = minimize(objective_function, x0=init_params, method='COBYLA', args=(G, backend, num_shots), constraints=cons)
            params = sol.x
            init_gamma = params[0::2]
            init_beta = params[1::2]
        qc = createCircuit_MaxCut(params, G, depth)
        temp_res_data = execute(qc, backend, shots=num_shots).result().results
        [E],_ = measurementStatistics_MaxCut(temp_res_data, G)
        if E>record:
            record = E
            record_params = params
    return record_params, record


def addWeights_MaxCut(G, decimals=0):
    """
    Adds weights G distributed from [0,1], rounded up to a number of decimals.
    Does not return anything, but modifies the input graph.
    :param G: The graph to modify.
    :param decimals: The number of decimals to use.
    """
    scaling_factor = np.power(10,decimals)
    for i,j in G.edges():
        w = np.ceil(np.random.uniform()*scaling_factor)/scaling_factor
        G.add_edge(i,j,weight=w)

def measurementStatistics_MaxCut(experiment_results, G):
    """
    Calculates the expectation and variance of the cost function. If
    results from multiple circuits are used as input, each circuit's
    expectation value are returned.
    :param experiment_results: Input on the form execute(...).result().results
    :param G: The graph on which the cost function is defined.
    :return: Lists of expectation values and variances
    """

    expectations = []
    variances = []
    num_qubits = G.number_of_nodes()
    for result in experiment_results:
        n_shots = result.shots
        counts = result.data.counts

        E = 0
        E2 = 0
        for hexkey in list(counts.__dict__.keys()):
            count = getattr(counts, hexkey)
            binstring = "{0:b}".format(int(hexkey,0)).zfill(num_qubits)
            binlist = [int(i) for i in binstring]
            cost = cost_MaxCut(binlist,G)
            E += cost*count/n_shots;
            E2 += cost**2*count/n_shots;

        if n_shots == 1:
            v = 0
        else:
            v = (E2-E**2)*n_shots/(n_shots-1)
        expectations.append(E)
        variances.append(v)
    return expectations, variances

def sampleUntilPrecision_MaxCut(circuit,G,backend,noisemodel,min_n_shots,max_n_shots,E_atol,E_rtol,dv_rtol,confidence_index):
    """
    Samples from the circuit and calculates the cost function until the specified
    error tolerances are satisfied. This may include several repetitions, either if
    the number of initial shots was too small, or if the variance estimate changed
    to a large degree since the last repetition, meaning that the required shot
    estimate was inaccurate.

    :param circuit: The circuit that will be sampled.
    :param G: The graph on which the cost function is defined.
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

        [E_cur],[v_cur] = measurementStatistics_MaxCut(experiment.result().results,G)
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
