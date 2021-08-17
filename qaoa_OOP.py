import os 
import sys

sys.path.append('../')
from qiskit_utilities.utilities import *
from qiskit import execute
from qiskit import transpile
from plots import * 

import numpy as np
import scipy.optimize as optimize

import itertools

class QAOABase:
    
    def __init__(self,options = None):
        self.options = options

        self.params        = dict()
        self.E             = dict()
        self.best          = dict()

        self.reset_bookkeeping_params()

    def reset_bookkeeping_params(self):
        
        self.g_it          = 0
        self.g_values      = dict()
        self.g_best_values = dict()
        self.g_params      = dict()
        

    # -----------------------------------------
    # Methods a child would have to implement:
    
    def initial_state(self):
        raise NotImplementedError
    
    def createCircuit(self):
        raise NotImplementedError

    def measurementStatistics(self):
        raise NotImplementedError

    def cost(self):
        raise NotImplementedError

    def getval(self):
        raise NotImplementedError

    # -----------------------------------------

    def get_depth_and_numCX(self):
        """
        Calculating the depth and number of cx-gates for full connectivity
        and 1 layer.
        
        Depth is here a bit ambiguous, but the returned depth referred to 
        denotes the length of the critical path of the circuit, not the depth 
        in the qaoa algorithm.

        Returns
        -------
        depth : int
            Length of the critical path of the one-layer circuit

        """
        # createCircuit modifies self.qc, so save this temporarily
        if hasattr(self, "qc"):
            circ_tmp   = self.qc
            temp_save  = True 
        else:
            temp_save  = False
 
        # create a circuit with self.q number of parameters, all ones
        # set usebarrier to False when calculating the depth
        usebarrier = self.options['usebarrier']
        self.options['usebarrier'] = False
        
        qc = self.createCircuit(np.ones(self.q))

        self.options['usebarrier'] = usebarrier
        
        # Transpiling circuit
        basis  = ['cx', 'id', 'rz', 'sx','x']
        new_qc = transpile(qc,basis_gates = basis, optimization_level=1)
        
        depth  = new_qc.depth()
        num_cx = new_qc.count_ops()['cx']

        # If the circuit was temporarily saved when making the circuit,
        # swap it back again. 
        if temp_save:
            self.qc = circ_tmp

        return depth, num_cx
            
    def interp_init(self):
        """
        Interpolates the current parameters to the next layer
        according to the procedure INTERP explained in 
        https://arxiv.org/pdf/1812.01041.pdf
    
        Returns
        -------
        x.flatten() : array
            array of length (self.depth)*(self.q) with the parameters for 
            the next depth of the algorithm.
        """

        x_prev = self.params[f'xL_d{self.depth - 1}']
        print(f"P = {x_prev}")

        # Minimize needs one-dimensional input, but the interpolation of the
        # parameters can be done with a multidimensional input
        #
        #    gamma_1 gamma_2 gamma_3 ...
        #    beta_1  beta_2  beta_3  ...
        #    ...
        # The resulting flattened vector will be
        #    gamma_1 beta_1 gamma-2 beta_2 ...
        # Which is what we want for the createCircuit method called in get-val

        # Hacky way of not having to refer to self.depth here
        p      = np.size(x_prev) // self.q
        
        x_prev = x_prev.reshape((p,self.q))
        x      = np.zeros((p+1,self.q))
       
        x[0,:] = x_prev[0,:]
        
        for i in range(2,p+1):
            x[i - 1,:] = (i - 1)/p  * x_prev[i-2,:] + (p - i + 1)/p * x_prev[i-1,:]
        
        x[p,:] = x_prev[p-1,:]
        print(f"P_ = {x.flatten()}")
        self.params[f'x0_d{self.depth}'] = x.flatten()
        
        return x.flatten()

    def save_best_params(self):

        # Find the best value along the path
        ind = max(self.g_values, key = self.g_values.get)

        # Save params
        
        self.params[f'xL_d{self.depth}'] = self.g_params[ind].copy()
        self.E[f'{self.depth}']          = self.g_values[ind]
        self.best[f'{self.depth}']       = self.g_best_values[ind]

    def get_energy_landscape(self):
        """
        Calculates the energy landscape given the simulation arguments
        specifying the number of parameters and their upper and lower limits.
        The method calls the scipy function brute which performs a grid search 
        over the specified ranges. The brute method also does some additional polishing 
        in the end.

        Returns
        -------
        Jout.T : array
            Array holding the energy at each point of the grid searched through
        x0.flatten() : array
            The parameters minimizing the energy found using brute force. 

        """

        print("Calculating energy landscape ...")

        ranges = []
        for i in range(self.q):

            step = (self.params_ul[i] - self.params_ll[i])/self.params_n[i]
            
            ranges.append(slice(self.params_ll[i],self.params_ul[i],step))
        
        x0, fval, grid, Jout = optimize.brute(self.getval, ranges, full_output = True)

        print("Calculating energy landscape done.")
        # Flatten the parameters according to fortran ordering
        self.params[f'x0_d{self.depth}'] = x0.flatten()
        
        # Transpose energy landscape to be in accordance with the ordering used in the
        # runQAOA function used previously.
        return Jout.T, x0.flatten()

    def simulate_init(self, **simulation_args):
        
        # Unpack the simulation arguments:
        
        self.backend     = simulation_args['backend']
        self.optmethod   = simulation_args['optmethod']
        self.max_depth   = simulation_args['max_depth']
        self.repeats     = simulation_args.get('repeats',1)
        self.shots       = simulation_args.get('shots',  1)
        self.noise_model = simulation_args.get('noise_model',None)

        # Give lower and upper limits (ll, ul) as arrays rather than
        # separately giving max and min for each variable, so that it is
        # easier to change the number of variables later on
        # Aslo, provide the number of steps for each variable as a similar array
        
        self.params_ll = simulation_args['params_ll']
        self.params_ul = simulation_args['params_ul']
        self.params_n  = simulation_args['params_n']

        self.q         = self.params_ll.size

        assert( self.params_ll.size == self.params_ul.size == self.params_n.size)

    def continue_simulation(self):
        """
        Boolean function, providing a critetion for continuing the simulation loop, 
        may be overridden by a child.

        """
        return self.depth <= self.max_depth
        
    def simulate(self, **simulation_args):

        """
        Simulation function for doing the QAOA algorithm.

        Parameters
        ----------
        simulation_args : dict
            Keyword arguments, must contain:
                backend
                optmethod
                max_depth
                params_ll : lower limits for each parameter to do brute search over
                params_ul : upper limits -- " --
                params_n  : number of points -- " --

                
        """

        self.simulate_init(**simulation_args)
        self.depth = 1
        
        # Calculate energy_landscape - global optimisation 

        Elandscape, x0 = self.get_energy_landscape()
        
        while self.continue_simulation():
           
                            
            # Reset the current book-keeping variables for each depth
            self.reset_bookkeeping_params()

            # Local optimisation

            for rep in range(self.repeats):
                print(f"Depth = {self.depth}, Rep = {rep + 1}")

                # No need to keep track of the optimisation result, as the getval-function
                # is required to update the member variables g_values, g_best_values, g_params
                # among all iterations, so that multiple repetitions can be performed and compared.
                # 
                # The function save_best_params() will ensure the best of each repetition will
                # be used further.
                
                _ = optimize.minimize(self.getval,
                                      x0 = x0,
                                      method = self.optmethod,
                                      options={'xatol': 1e-2, 'fatol': 1e-1, 'disp': True})
            
            self.save_best_params()
            self.depth += 1
            
            # Extrapolate the parameters to the next depth
            x0 = self.interp_init()

        return Elandscape, self.params, self.E, self.best
            
            
class QAOAStandard(QAOABase):

    def __init__(self,qubits,options = None):
        super().__init__(options)
        
        # If a start circuit is provided, use this in self.initial_state(qubits)
        self.start_circuit = options.get('start_circuit', None)
        self.qubits = qubits

    def simulate_init(self, **simulation_args):
        super().simulate_init(**simulation_args)

        # Only create state strings array if using the statevector simulation
        if "statevector" in self.backend.name().split('_'):
            self.generate_state_strings(self.qubits)
        
    def generate_state_strings(self, qubits):
        """
        Generates an array of all the state strings in increasing order. 
        Useful for toy examples.

        This should only be done when using the state-vector simulator,
        so do a call to this function in the simulate_init function.
        
        Parameters
        ----------
        qubits : int 
            number of qubits in circuit.

        """

        self.state_strings = np.array([''.join(i) for i in itertools.product('01', repeat= qubits)])        
        # Flip the state-vectors to be in accordance with the ordering in qiskit:
        #self.state_strings = map(lambda x : x[::-1], self.state_strings)

    def initial_state(self, qubits):
        # Save current circuit and registers as
        # member variables of the object
        # Use |+>^n as initial state

        q = QuantumRegister(qubits)
        c = ClassicalRegister(qubits)
        qc = QuantumCircuit(q,c, name = self.options.get('name', None))

        # If provided with a start_circuit, use this instead of |+>^n
        if self.start_circuit is not None:
            qc.compose(self.start_circuit, inplace = True)
        else:
            qc.h(range(qubits))

        self.q_register = q
        self.c_register = c
        self.qc         = qc

    def getval(self, params):
        """
        Objective function to use in the minimizer. Saves in each iteration the 
        parameters and the function values in the member variables whose name starts 
        with g_ , i.e. what previously was the global book-keeping variables.

        Parameters
        ----------
        params : array
            Array of parameters to feed into the circuit: gamma_1, beta_1, ... , beta_p
            or even more parameters per. layer.
        
        Returns
        -------
        -val[0] : float
            Value of hamiltonian for these parameters. 

        """
        

        self.g_it += 1

        circuit = self.createCircuit(params)

        if self.backend.configuration().local:
            job = execute(circuit,
                          backend = self.backend,
                          noise_model = self.noise_model,
                          shots = self.shots)
        else:
            job = start_or_retrieve_job(name +"_"+str(g_it),
                                        self.backend,
                                        circuit,
                                        options = {'shots' : self.shots})

        val, _, bval = self.measurementStatistics(job)

        self.g_values[str(self.g_it)]      = val[0]
        self.g_best_values[str(self.g_it)] = bval
        self.g_params[str(self.g_it)]      = params

        return -val[0]
