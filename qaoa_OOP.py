import os 
import sys

sys.path.append('../')
from qiskit_utilities.utilities import *
from qiskit import execute
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
    
    def interp_init(self):

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
        
        while self.depth < self.max_depth:
           
                            
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
        self.generate_state_strings(qubits) 
        
    def generate_state_strings(self, qubits):
        """
        Generates an array of all the state strings in increasing order. 
        Useful for toy examples.
        
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




