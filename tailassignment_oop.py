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

            self.qc.rz( gamma * hr, self.q_register[r])

            for r_ in range(r+1,self.R):
                Jrr_  = 0.5 * self.FR[:,r] @ self.FR[:,r_]

                # Apply U(gamma), coupling part

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

            self.qc.rz(gamma * hr, self.q_register[r])

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

            self.qc.rz( gamma * hr, self.q_register[r])

            for r_ in range(r+1,self.R):
                Jrr_  = 0.5 * self.FR[:,r] @ self.FR[:,r_]

                # Apply U(gamma), coupling part

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

        for d in range(self.depth):

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
        """

        SP = np.zeros(self.max_depth)
        C  = np.zeros(self.max_depth)
        
        self.depth = 1
        while self.depth <= self.max_depth:
        
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

            else:

                counts            = job.result().get_counts()
                binstrings        = np.array(list(counts.keys()))
                counts_per_string = np.array(list(counts.values()))

                C[self.depth - 1] = self.vector_cost(binstrings) @ counts_per_string / self.shots

            self.depth += 1
        if plot:
            plot_H_prob(self,SP,C, savefig)

        return SP, C

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
        # force the weights to be 0. 

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

class TailAssignmentSepDisc(QAOATailAssignment):

    def __init__(self, options):
        super().__init__(options)

        # The separated discretization needs a threshold for
        # knowing when to swap hamiltonian

        self.tol      = options.get("tol", 0.9)
        self.swap     = False

        # Boolean to keep track of whether the energy landscape is recalculated
        self.recalc_E = False

    def cost(self, binstring):

        # Use only the exact cover cost before swapping the hamiltonian.
        # Reverse string since qiskit uses ordering MSB ... LSB
        x = np.array(list(map(int,binstring[::-1])))
        
        if self.swap:
            return - ( (self.CR @ x) +  self.mu * np.sum((1 - (self.FR @ x))**2) )
        else:   
            return - ( self.mu * np.sum((1 - (self.FR @ x))**2) )
        
        
    def initial_state(self):

        if not self.swap:
            super(TailAssignmentSepDisc,self).initial_state(self.R)

        else:
            # If time to change, set the initial
            # circuit to the exact cover circuit

            # Check whether I have messed up saving qc_exco
            if not hasattr(self, "qc_exco"):
                print("Error, the exact cover circuit is not saved yet.")
                raise AttributeError

            q = QuantumRegister(self.R)
            c = ClassicalRegister(self.R)
            
            self.qc = QuantumCircuit(q,c, name = self.options.get('name', None))
            self.q_register = q
            self.c_register = c

            # Add the exact cover part of the circuit to the beginning
            # inplace = True ensures that the result is saved in self.qc
            self.qc.compose(self.qc_exco.copy(), inplace = True)

            
    def mix_states(self,beta):
        if not self.swap:
            super().mix_states(beta)
        else:
            
            # If time to change, hamiltonian, make
            # sure to also change the mixer accordingly

            # Try different variants here
            
            super().mix_states( + beta)
            self.qc.compose(self.qc_exco.copy(), inplace = True)
            super().mix_states( - beta)
            #self.qc.compose(self.qc_exco, inplace = True)

    def new_depth_init(self):
        # Do a test of whether to swap the hamiltonian:

        job = execute(self.qc,
                      backend = self.backend,
                      noise_model = self.noise_model,
                      shots = self.shots)
            
        sp = self.successProbability(job)

        if sp >= self.tol and not self.swap:
            self.swap    = True
            
            # Save starting part of circuit in object
            self.qc_exco = self.qc

            # Save theta parameters - might be used in mixer
            # in the same fashion as the warm start mixer

            # Currently not in use in the mixer
            
            if "statevector" in self.backend.name().split('_'):

                statevector = job.result().get_statevector()
                probs = np.abs(statevector)**2
                states = np.array([list(map(int,s)) for s in self.state_strings])

                self.thetas = np.arccos(probs @ states)
                
            else:
                # not implemented for other simulators
                pass    
            

    def interp_init(self):
        """
        Override this function to include the 
        calculation of the energy landscape when 
        the hamiltonian is swapped.
        """

        self.new_depth_init()

        if self.swap and not self.recalc_E:
            _, x0 = self.get_energy_landscape()
            self.recalc_E = True
            return x0
        
        else:
            return super().interp_init()
            
            
    def createCircuit(self, params):

        """
        Implements the ciruit for the tail assignment problem, with 'separated discretization'

        Parameters
        ----------
        params : array
            variational parameters gamma_1 beta_1 , ... , gamma_p beta_p

        Returns
        -------
        qc : QuantumCircuit    

        """
        
        self.initial_state()
        
        gammas = params[::2]
        betas  = params[1::2]

        D      = np.size(gammas)
        
        for d in range(D):

            gamma = gammas[d]
            beta  = betas[d]

            if self.swap:
                # Use complete hamiltonian or just cost here?
                # self.apply_cost(gamma)
                self.apply_hamiltonian(gamma)
            else:
                self.apply_exco(gamma)
            
            # Apply mixer U(beta) at the end
            # Mixer will depend on self.swap
            
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
        
        This function must be modified for the separated discretization since 
        the swap parameter has to be reset.

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
        """

        SP = np.zeros(self.max_depth)
        C  = np.zeros(self.max_depth)

        # Reset the swap parameter when calculating the statistics
        self.swap = False
        
        self.depth = 1

        # Save the _total_ cost of the current state - should not be dependent on
        # the swap variable, as it will make the comparison weird in the plot

        tot_cost = np.vectorize(super().cost)
        
        while self.depth <= self.max_depth:
        
            qc  = self.createCircuit(self.params[f'xL_d{self.depth}'])
            job = execute(qc,
                          backend = self.backend,
                          noise_model = self.noise_model,
                          shots = self.shots)
            
            sp = self.successProbability(job)
            SP[self.depth - 1] = sp
            
            if sp >= self.tol:
                
                self.swap = True
            
            if "statevector" in self.backend.name().split('_'):
                
                statevector = job.result().get_statevector()
                probs = (np.abs(statevector))**2
                
                C[self.depth - 1 ] = tot_cost(self.state_strings) @ probs

            else:

                counts            = job.result().get_counts()
                binstrings        = np.array(list(counts.keys()))
                counts_per_string = np.array(list(counts.values()))
                
                C[self.depth - 1] = tot_cost(binstrings) @ counts_per_string / self.shots

            self.depth += 1
            
        if plot:
            plot_H_prob(self,SP,C, savefig)

        return SP, C

