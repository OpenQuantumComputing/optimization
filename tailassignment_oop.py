from qaoa_OOP import *

class QAOATailAssignment(QAOAStandard):

    def __init__(self,options):
        number_of_qubits = options['CR'].size 
        super().__init__(number_of_qubits, options)

        # Vectorized functions for the cost and is_solution:
        # Very useful for simulation with the statevector, as it
        # allows for avoiding multiple nested loops in the measurementStatistics function
        
        self.vector_cost = np.vectorize(self.cost)
        self.vector_is_solution = np.vectorize(self.is_solution)

        # Save matrices as variables of object:
        
        self.FR         = self.options.get('FR', None)
        self.CR         = self.options.get('CR', None)
        self.mu         = self.options.get('mu', 1)

        self.F, self.R  = np.shape(self.FR)
    
    def cost(self,binstring):
        
        # Reverse string since qiskit uses ordering MSB ... LSB
        x = np.array(list(map(int,binstring[::-1])))
        return - ( (self.CR @ x) + self.mu * np.sum((1 - (self.FR @ x))**2) )

    def mix_states(self,beta):
        """
        Applies unitary evolution of mixer hamiltonian with 
        time parameter beta

        Parameters
        ----------
        beta : float
            Time/angle for applying hamiltonian.

        """
        self.qc.rx( - 2 * beta, self.q_register ) 

    def apply_exco(self,gamma):
        """
        Applies unitary evolution of exact cover hamiltonian with 
        time-parameter gamma

        Parameters
        ----------
        gamma : float
            Time/angle for applying hamiltonian.

        """
        
        for r in range(self.R):
            hr = 0.5 * self.mu * self.FR[:,r] @ (np.sum(self.FR,axis = 1) - 2)

            # Apply only if hr differs sufficiently from 0
            if not np.isclose(hr, 0):
                self.qc.rz( gamma * hr, self.q_register[r])

            for r_ in range(r+1,self.R):
                Jrr_  = 0.5 * self.FR[:,r] @ self.FR[:,r_]

                # Apply U(gamma), coupling part
                # Only if the coupling constant differs sufficiently from 0

                if not np.isclose(Jrr_, 0):
                    self.qc.cx(self.q_register[r], self.q_register[r_])
                    self.qc.rz(gamma * Jrr_, self.q_register[r_])
                    self.qc.cx(self.q_register[r], self.q_register[r_])

                    
    def apply_cost(self,gamma):
        """
        Applies unitary evolution of cost hamiltonian with 
        time-parameter gamma

        Parameters
        ----------
        gamma : float
            Time/angle for applying hamiltonian.

        """

        for r in range(self.R):
            hr = 0.5 * self.CR[r]
            
            # Apply only if hr differs sufficiently from 0
            if not np.isclose(hr, 0):
                self.qc.rz( gamma * hr, self.q_register[r])

    def apply_hamiltonian(self,gamma):
        """
        Applies unitary evolution of the full hamiltonian representing the 
        problem, with time parameter gamma

        Parameters
        ----------
        gamma : float
            Time/angle for applying hamiltonian.

        """
        
        for r in range(self.R):
            hr = 0.5 * self.CR[r] + 0.5 * self.mu * self.FR[:,r] @ (np.sum(self.FR,axis = 1) - 2)

            # Apply only if hr differs sufficiently from 0
            if not np.isclose(hr, 0):
                self.qc.rz( gamma * hr, self.q_register[r])

            for r_ in range(r+1,self.R):
                Jrr_  = 0.5 * self.FR[:,r] @ self.FR[:,r_]

                # Apply U(gamma), coupling part

                # Apply U(gamma), coupling part
                # Only if the coupling constant differs sufficiently from 0

                if not np.isclose(Jrr_, 0):
                    self.qc.cx(self.q_register[r], self.q_register[r_])
                    self.qc.rz(gamma * Jrr_, self.q_register[r_])
                    self.qc.cx(self.q_register[r], self.q_register[r_])

                
    def createCircuit(self, params):

        """
        Implements the ciruit for the tail assignment problem
        Parameters
        ----------
        params : array
            variational parameters gamma_1 beta_1, ... , gamma_p beta_p

        Returns
        -------
        qc : QuantumCircuit    

        """

        self.initial_state(self.R)
        
        gammas = params[::2]
        betas  = params[1::2]

        D = np.size(gammas)

        for d in range(D):

            gamma = gammas[d]
            beta  = betas[d]

            # Hamiltonian - cost + constraint
            self.apply_hamiltonian(gamma)
            # This is an equivalent implementation, but requires more gates.
            # as the h-terms are not collected together
            #self.apply_cost(gamma)
            #self.apply_exco(gamma)

            if self.options['usebarrier']:
                self.qc.barrier()

            # Apply mixer U(beta):
            self.mix_states(beta)
            
            if self.options['usebarrier']:
                self.qc.barrier()
                
        if "statevector" not in self.backend.name().split('_'):
            # Do not measure at the end of the circuit if using a
            # statevector simulation 
            self.qc.measure(self.q_register,self.c_register)

        return self.qc

    def simulation_statistics(self, best_sol = None , plot = True, savefig = None):
        """
        Do simulation again with optimal found parameters and 
        return the success probability together with the average hamiltonian.

        Parameters
        ----------
        plot : bool
            If True, plot the results
        savefig : string or None
            If not None, save the figure with the filename/path given by savefig 

        Returns
        -------
        SP : array
            Success probability as a function of depth
        C : array
            Total cost as a function of depth.
        P : array
            Probability of best state as a function of depth
        """

        SP = np.zeros(self.max_depth)
        C  = np.zeros(self.max_depth)
        P  = np.zeros(self.max_depth)

        # Provide the optimal solution if not running a statevector simulation
        # The simulation statistics depends on either using the statevector simulator
        # or explicitly providing the best solution as an argument to this function.
        
        assert("statevector" in self.backend.name().split('_') or best_sol != None)
        
        if best_sol == None:
            best_index = np.argmax( self.vector_cost(self.state_strings) )
            best_sol   = self.state_strings[best_index]
            
        self.depth = 1
        while self.continue_simulation():
        
            qc  = self.createCircuit(self.params[f'xL_d{self.depth}'])
            job = execute(qc,
                          backend = self.backend,
                          noise_model = self.noise_model,
                          shots = self.shots)
            
            SP[self.depth - 1] = self.successProbability(job)
            
            if "statevector" in self.backend.name().split('_'):
                
                statevector = job.result().get_statevector()
                probs = (np.abs(statevector))**2
                        
                C[self.depth - 1 ] = self.vector_cost(self.state_strings) @ probs
                P[self.depth - 1 ] = probs[best_index]
            else:

                counts            = job.result().get_counts()
                binstrings        = np.array(list(counts.keys()))
                counts_per_string = np.array(list(counts.values()))

                C[self.depth - 1] = self.vector_cost(binstrings) @ counts_per_string / self.shots

                if best_sol not in binstrings:
                    P[self.depth - 1] = 0
                else:
                    P[self.depth - 1] = counts_per_string[ binstrings == best_sol ][0] / self.shots
        
            self.depth += 1
        if plot:
            plot_H_prob(self,SP,C, savefig)

        return SP, C, P 

    def is_solution(self,binstring):
        a = np.array(list(map(int,binstring[::-1])))
        return np.all(np.sum(self.FR * a,1)-1 == 0)

    def successProbability(self,job):
        """
        Calculation of the success probability for a given job
        Separating the calculation for a simulation done with the 
        statevector is a bit hackey.

        Parameters
        ----------
        job : Qiskit job
           
        Returns
        -------
        s_prob : float
            Success probability 
        """
        
        
        if "statevector" in self.backend.name().split('_'):
            
            experiment_results = job.result().results
            statevector = job.result().get_statevector()
            probs = (np.abs(statevector))**2

            s_prob = self.vector_is_solution(self.state_strings) @ probs
            
        else:
            
            experiment_results = job.result().results
            s_prob = 0
            # assumes experiment results is a one-dimensional array

            for result in experiment_results:
                n_shots = result.shots
                counts = result.data.counts
                for hexkey in list(counts.keys()):
                    count = counts[hexkey]
                    binstring = "{0:b}".format(int(hexkey,0)).zfill(self.R)
                    if self.is_solution(binstring):
                        s_prob += count/n_shots
            
        return s_prob

    def measurementStatistics(self,job):
        """
        Measurement statistics for the tail assignment problem.
        Calculating only the expectations and the best cost

        Parameters
        ----------
        job : Qiskit job
            ...

        Returns
        -------
            expectation_values : array
            variances          : array
            cost_best          : float 

        """

        cost_best = - np.inf
        experiment_results = job.result().results
        expectations = np.zeros(len(experiment_results))

        ## Simulation done with statevector
        ## Much faster than doing it with other simulators in this implementation,
        ## as many operations are  vectorized here and don't depend on iterating through
        ## dictionaries
        
        if "statevector" in self.backend.name().split('_'):

            statevector = job.result().get_statevector()
            probs = np.abs(statevector)**2

            costs = self.vector_cost(self.state_strings)
            E     = costs @ probs

            best_sampled_state = np.max( costs @ np.ceil(probs))
            cost_best          = max(cost_best,best_sampled_state)

            expectations[0] = E 

        ## Simulation done with other simulators
        
        else:
            
           for i, result in enumerate(experiment_results):
               n_shots = result.shots
               counts = result.data.counts

               E  = 0

               for hexkey in list(counts.keys()):
                   count     = counts[hexkey]
                   binstring = "{0:b}".format(int(hexkey,0)).zfill(self.R)
                   cost      = self.cost(binstring)
                   cost_best = max(cost_best, cost)
                   E        += cost*count/n_shots

               expectations[i] = E
               

        return expectations, None , cost_best

class QAOAExactCover(QAOATailAssignment):

    def __init__(self, options = None):

        # Exactly the same as the tail assignment class, except we
        # force the weights to be 0, even if non-zero weights are provided 

        CR = np.zeros_like(options['FR'][0,:])
        options['CR'] = CR

        self.tol = options.get('tol', 0.9)
        
        super().__init__(options)

    def cost(self, binstring):
        # Reverse string since qiskit uses ordering MSB ... LSB
        x = np.array(list(map(int,binstring[::-1])))
        return - (  self.mu * np.sum((1 - (self.FR @ x))**2) )
        
    def apply_hamiltonian(self,gamma):
        # The hamiltonian in this case is only the exact cover part
        super().apply_exco(gamma)

    def continue_simulation(self):
        """
        Terminate the optimisation loop if the success probability of 
        the previous iteration is higher than the tolerance provided.
        If the depth exceeds max_depth, then terminate anyways.

        """
        if self.depth <= self.max_depth:
            job = execute(self.qc,
                          backend = self.backend,
                          noise_model = self.noise_model,
                          shots = self.shots)

            sp = self.successProbability(job)

            return sp < self.tol
        else:
            return False
        

class TailAssignmentInterlaced(QAOATailAssignment):

    def createCircuit(self, params):

        """
        Implements the ciruit for the tail assignment problem, with three parameters
        Parameters
        ----------
        params : array
            variational parameters gamma_1 beta_1 delta_1 , ... , gamma_p beta_p delta_p

        Returns
        -------
        qc : QuantumCircuit    

        """

        self.initial_state(self.R)
        
        gammas = params[::3]
        betas  = params[1::3]
        deltas = params[2::3]

        D      = np.size(gammas)

        for d in range(D):

            gamma = gammas[d]
            beta  = betas[d]
            delta = deltas[d]

            # Hamiltonian - weights 
            self.apply_cost(delta)

            if self.options['usebarrier']:
                self.qc.barrier()

             # Apply mixer U(beta) inbetween hamiltonians
            self.mix_states(beta)

            if self.options['usebarrier']:
                self.qc.barrier()

            self.apply_exco(gamma)

            # Apply mixer U(beta) at the end
            self.mix_states(beta)
            
            if self.options['usebarrier']:
                self.qc.barrier()
                
        if "statevector" not in self.backend.name().split('_'):
            # Do not measure at the end of the circuit if using a
            # statevector simulation 
            self.qc.measure(self.q_register,self.c_register)

        return self.qc




class TailAssignmentNFam(QAOATailAssignment):
    """
    This class implements the freedom for the QAOA to rotate each qubit around a different axis in the xy-plane when mixing.
    The axis is decided by one variational parameter per qubit which is optimized along with the other variational parameters.
    For all layers at a given depth the qubit rotates around the same axis.
    This is the NFam from https://arxiv.org/abs/2107.13129.
    """

    def __init__(self,options):
        super().__init__(options)

        #Need an initial axis for the mixer to rotate around
        #This can be chosen in options, or defaults to the (1,1) axis
        self.init_thetas = options.get('init_thetas', np.zeros(self.FR.shape[1])+np.pi/4)
    def mix_states(self,beta, thetas):
        """
        Applies unitary evolution of mixer hamiltonian with
        time parameter beta

        Parameters
        ----------
        beta : float
            Time/angle for applying hamiltonian.
        thetas : array of floats
            Angles deciding which axis to rotate around

        """
        for i in range(self.FR.shape[1]):
            if np.abs(2 * beta * np.cos(thetas[i]))>1e-6:
                self.qc.rx( - 2 * beta * np.cos(thetas[i]), self.q_register[i] )
            if np.abs(2 * beta * np.sin(thetas[i]))>1e-6:
                self.qc.ry( - 2 * beta * np.sin(thetas[i]), self.q_register[i] )


    def createCircuit(self, params):

        """
        Implements the ciruit for the tail assignment problem, with two parameters
        and an extra number of params equvalent to the number of qubits
        Parameters
        ----------
        params : array
            variational parameters theta_1, ...theta_n, gamma_1 beta_1 , ... , gamma_p beta_p
            except for in the first layer when it only contains gamma_1 beta_1 to allow for a grid search

        Returns
        -------
        qc : QuantumCircuit

        """

        self.initial_state(self.R)
        rN = self.FR.shape[1]
        if self.depth==1:
            #Sets the correct params, and the initial theta values
            gammas = params[::2]
            betas  = params[1::2]
            thetas = self.init_thetas
        else:
            #Picks out the rotation angles first and the sets the rest of the params as usual
            thetas = params[:rN]
            gammas = params[rN::2]
            betas  = params[rN + 1::2]

        for d in range(self.depth):

            gamma = gammas[d]
            beta  = betas[d]

            # Hamiltonian - cost + constraint
            self.apply_hamiltonian(gamma)

            if self.options['usebarrier']:
                self.qc.barrier()

            # Apply mixer U(beta, thetas) at the end
            self.mix_states(beta, thetas)

            if self.options['usebarrier']:
                self.qc.barrier()

        if "statevector" not in self.backend.name().split('_'):
            # Do not measure at the end of the circuit if using a
            # statevector simulation
            self.qc.measure(self.q_register,self.c_register)

        return self.qc

    def interp_init(self):
        x_prev_0 = self.params[f'xL_d{self.depth - 1}']
        print(f"P = {x_prev_0}")
        rN = self.FR.shape[1]

        # Minimize needs one-dimensional input, but the interpolation of the
        # parameters can be done with a multidimensional input
        #
        #    gamma_1 gamma_2 gamma_3 ...
        #    beta_1  beta_2  beta_3  ...
        #    ...
        # The resulting flattened vector will be
        #    theta_1, .. theta_n, gamma_1 beta_1 gamma-2 beta_2 ...
        # Which is what we want for the createCircuit method called in get-val

        p      = self.depth - 1
        if self.depth > 2:
            #Updates all params according to heuristic, except the theta values as they are not part of the heuristic.
            #The theta values are the first rN values
            x_prev = x_prev_0[rN:].reshape((p,self.q))
        else:
            #For the first update the theta values need to be added seperately
            x_prev_0 = self.init_thetas
            x_prev = self.params[f'xL_d{self.depth - 1}']
        x      = np.zeros((p+1,self.q))
        x_prev = x_prev.reshape((p,self.q))
        x[0,:] = x_prev[0,:]

        for i in range(2,p+1):
            x[i - 1,:] = (i - 1)/p  * x_prev[i-2,:] + (p - i + 1)/p * x_prev[i-1,:]

        x[p,:] = x_prev[p-1,:]
        print(f"P_ = {x.flatten()}")
        self.params[f'x0_d{self.depth}'] = x.flatten()

        return np.append(x_prev_0[:rN], x.flatten())


class TailAssignmentInterlacedNFam(TailAssignmentNFam):
    """
    This class extends the NFam from https://arxiv.org/abs/2107.13129 to also include our interlaced method.
    """


    def createCircuit(self, params):

        """
        Implements the ciruit for the tail assignment problem, with three parameters
        and an extra number of params equvalent to the number of qubits
        Parameters
        ----------
        params : array
            variational parameters theta_1, ..., theta_n gamma_1 beta_1 delta_1 , ... , gamma_p beta_p delta_p

        Returns
        -------
        qc : QuantumCircuit

        """

        self.initial_state(self.R)
        rN = self.FR.shape[1]
        if self.depth==1:
            #Sets the correct params, and the initial theta values
            gammas = params[::3]
            betas  = params[1::3]
            deltas = params[2::3]
            thetas = self.init_thetas
        else:
            #Picks out the rotation angles first and the sets the rest of the params as usual
            thetas = params[:rN]
            gammas = params[rN::3]
            betas  = params[rN + 1::3]
            deltas  = params[rN + 2::3]


        for d in range(self.depth):

            gamma = gammas[d]
            beta  = betas[d]
            delta = deltas[d]

            # Hamiltonian - weights
            self.apply_cost(delta)

            if self.options['usebarrier']:
                self.qc.barrier()

             # Apply mixer U(beta, thetas) inbetween hamiltonians
            self.mix_states(beta, thetas)

            if self.options['usebarrier']:
                self.qc.barrier()

            # Hamiltonian - constraints
            self.apply_exco(gamma)

            if self.options['usebarrier']:
                self.qc.barrier()
            # Apply mixer U(beta, thetas) at the end
            self.mix_states(beta, thetas)

            if self.options['usebarrier']:
                self.qc.barrier()

        if "statevector" not in self.backend.name().split('_'):
            # Do not measure at the end of the circuit if using a
            # statevector simulation
            self.qc.measure(self.q_register,self.c_register)

        return self.qc
