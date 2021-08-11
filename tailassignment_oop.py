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

    def simulation_statistics(self, plot = True, savefig = None):
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
        
        best_sol = np.argmax( self.vector_cost(self.state_strings) )
        
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
                P[self.depth - 1 ] = probs[best_sol]
            else:

                counts            = job.result().get_counts()
                binstrings        = np.array(list(counts.keys()))
                counts_per_string = np.array(list(counts.values()))

                C[self.depth - 1] = self.vector_cost(binstrings) @ counts_per_string / self.shots

                if self.state_strings[best_sol] not in binstrings:
                    P[self.depth - 1] = 0
                else:
                    P[self.depth - 1] = counts_per_string[binstrings == self.state_string[best_sol]] / self.shots
        
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
                    binstring = "{0:b}".format(int(hexkey,0)).zfill(rN)
                    if is_Solution(binstring, FR):
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
