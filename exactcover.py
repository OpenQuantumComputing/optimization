from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import numpy as np

#import os
#import sys
#sys.path.append('../')
#from qiskit_utilities.utilities import *

def createCircuit_ExactCover(x, depth, options=None):
    """
    implements https://arxiv.org/pdf/1912.10499.pdf
    FR is a matrix 
        r1 r2 ..... rN
    f1
    f2
    .
    .
    .
    fn
    where each column indicates if route r uses flight f (entry is 1) or not (entry is 0)
    
    CR is a vector of route costs [c1 ... cN] where c1 is the cost of route r1 
    """

    FR = options.get('FR', None)
    CR = options.get('CR', None)
    usebarrier = options.get('usebarrier', False)
    name = options.get('name', None)

    fn=FR.shape[0]### number of flights
    rN=FR.shape[1]### number of routes

    q = QuantumRegister(rN)
    c = ClassicalRegister(rN)
    circ = QuantumCircuit(q, c, name=name)
    circ.h(range(rN))

    for d in range(depth):
        gamma = x[2 * d]
        beta = x[2 * d + 1]
        for i in range(rN):
            w=0
            for j in range(fn):
                w += .5*FR[j,i]*(np.sum(FR[j,:])-2)
            if CR is not None:
                w += .25*CR[i]**2
            if abs(w)>1e-14:
                wg = w * gamma
                circ.rz(wg, q[i])
            ###
            for j in range(i+1, rN):
                w=0
                for k in range(fn):
                    w += 0.5*FR[k,i]*FR[k,j]
                if CR is not None:
                    if (i == j):
                        w += 0.25*CR[i]**2
                if w>0:
                    wg = w * gamma
                    circ.cx(q[i], q[j])
                    circ.rz(wg, q[j])
                    circ.cx(q[i], q[j])
                    # this is an equivalent implementation:
                    #    circ.cu1(-2 * wg, i, j)
                    #    circ.u1(wg, i)
                    #    circ.u1(wg, j)
                    if usebarrier:
                        circ.barrier()

        circ.rx(-2 * beta, range(rN))
        if usebarrier:
            circ.barrier()
    circ.measure(q, c)
    return circ

def cost_exactCover(binstring, FR, CR):
    rN=FR.shape[1]### number of routes
    a=np.zeros(rN)
    for i in np.arange(len(binstring)):
        ### inverse order, because qiskit order is $q_n q_{n-1} .... q_0$
        a[len(binstring)-i-1]=int(binstring[i])
    if CR is None:
        return -np.sum((np.sum(FR*a,1) -1)**2)
    else:
        return -np.sum((np.sum(FR*a,1) -1)**2) - np.sum(a*(CR**2))

def measurementStatistics_ExactCover(experiment_results, options=None):
    """
    Calculates the expectation and variance of the cost function. If
    results from multiple circuits are used as input, each circuit's
    expectation value is returned.
    :param experiment_results: Input on the form execute(...).result().results
    :param G: The graph on which the cost function is defined.
    :return: Lists of expectation values and variances
    """

    FR = options.get('FR', None)
    CR = options.get('CR', None)
    rN=FR.shape[1]### number of routes

    cost_best = -np.inf

    expectations = []
    variances = []
    for result in experiment_results:
        n_shots = result.shots
        counts = result.data.counts

        E = 0
        E2 = 0
        for hexkey in list(counts.keys()):
            count = counts[hexkey]
            binstring = "{0:b}".format(int(hexkey,0)).zfill(rN)
            cost = cost_exactCover(binstring, FR, CR)
            cost_best = max(cost_best, cost)
            E += cost*count/n_shots
            E2 += cost**2*count/n_shots

        if n_shots == 1:
            v = 0
        else:
            v = (E2-E**2)*n_shots/(n_shots-1)
        expectations.append(E)
        variances.append(v)
    return expectations, variances, cost_best


def is_Solution(binstring, FR):
    rN=FR.shape[1]### number of routes
    fn=FR.shape[0]### number of flights
    a=np.zeros(rN)
    for i in np.arange(len(binstring)):
        ### inverse order, because qiskit order is $q_n q_{n-1} .... q_0$
        a[len(binstring)-i-1]=int(binstring[i])
    return np.sum(np.sum(FR*a,1)-1)==0


def successProbability(experiment_results, options=None):

    FR = options.get('FR', None)
    CR = options.get('CR', None)
    rN=FR.shape[1]

    success_prob = []

    for result in experiment_results:
        n_shots = result.shots
        counts = result.data.counts

        sprop=0
        for hexkey in list(counts.keys()):
            count = counts[hexkey]
            binstring = "{0:b}".format(int(hexkey,0)).zfill(rN)
            if is_Solution(binstring, FR):
                sprop+=count

        success_prob.append(sprop/n_shots)
    return success_prob
