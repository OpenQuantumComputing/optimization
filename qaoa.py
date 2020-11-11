from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
import numpy as np
import networkx as nx
import math
import os
from scipy.optimize import minimize

import sys
sys.path.append('../')

from qiskit_utilities.utilities import *

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

def Cn_U3_0theta0(qc, control_indices, target_index, theta):
    """
    Ref: https://arxiv.org/abs/0708.3274

    """
    n=len(control_indices)
    if n == 0:
        qc.rz(theta, control_indices)
    elif n == 1:
        qc.cu3(0, theta, 0, control_indices, target_index)
    elif n == 2:
        qc.cu3(0, theta/ 2, 0, control_indices[1], target_index)  # V gate, V^2 = U
        qc.cx(control_indices[0], control_indices[1])
        qc.cu3(0, -theta/ 2, 0, control_indices[1], target_index)  # V dagger gate
        qc.cx(control_indices[0], control_indices[1])
        qc.cu3(0, theta/ 2, 0, control_indices[0], target_index) #V gate
    else:
        raise Exception("C^nU_3(0,theta,0) not yet implemented for n="+str(n)+".")

def CGp(qc, control_index, target_index, p):
    """
    Ref: https://onlinelibrary.wiley.com/doi/pdf/10.1002/qute.201900015

    """
    thetadash = np.arcsin(np.sqrt(p))
    qc.u(thetadash, 0, 0, target_index)
    qc.cx(control_index, target_index)
    qc.u(-thetadash, 0, 0, target_index)

def Wn(qc, indices):
    """
    Ref: https://onlinelibrary.wiley.com/doi/pdf/10.1002/qute.201900015

    """
    n=len(indices)
    if n<2 or n>8:
        raise Exception("Wn not defined for n="+str(n)+".")

    qc.x(indices[0])

    if n==2:
        qc.h(indices[1])
        qc.cx(indices[1], indices[0])
    elif n==3:
        CGp(qc, indices[0], indices[1], 1/3)
        qc.cx(indices[1], indices[0])
        #
        CGp(qc, indices[1], indices[2], 1/2)
        qc.cx(indices[2], indices[1])
    elif n==4:
        CGp(qc, indices[0], indices[1], 1/4)
        qc.cx(indices[1], indices[0])
        #
        CGp(qc, indices[1], indices[2], 1/3)
        qc.cx(indices[2], indices[1])
        #
        CGp(qc, indices[2], indices[3], 1/2)
        qc.cx(indices[3], indices[2])
    elif n==5:
        CGp(qc, indices[0], indices[1], 2/5)
        qc.cx(indices[1], indices[0])
        #
        CGp(qc, indices[0], indices[2], 1/2)
        qc.cx(indices[2], indices[0])
        #
        CGp(qc, indices[1], indices[3], 1/3)
        qc.cx(indices[3], indices[1])
        #
        CGp(qc, indices[3], indices[4], 1/2)
        qc.cx(indices[4], indices[3])
    elif n==6:
        CGp(qc, indices[0], indices[1], 3/6)
        qc.cx(indices[1], indices[0])
        #
        CGp(qc, indices[0], indices[2], 1/3)
        qc.cx(indices[2], indices[0])
        #
        CGp(qc, indices[1], indices[3], 2/3)
        qc.cx(indices[3], indices[1])
        #
        CGp(qc, indices[2], indices[4], 1/2)
        qc.cx(indices[4], indices[2])
        #
        CGp(qc, indices[1], indices[5], 1/2)
        qc.cx(indices[5], indices[1])
    elif n==7:
        CGp(qc, indices[0], indices[1], 3/7)
        qc.cx(indices[1], indices[0])
        #
        CGp(qc, indices[0], indices[2], 1/3)
        qc.cx(indices[2], indices[0])
        #
        CGp(qc, indices[1], indices[3], 1/2)
        qc.cx(indices[3], indices[1])
        #
        CGp(qc, indices[2], indices[4], 1/2)
        qc.cx(indices[4], indices[2])
        #
        CGp(qc, indices[1], indices[5], 1/2)
        qc.cx(indices[5], indices[1])
        #
        CGp(qc, indices[3], indices[6], 1/2)
        qc.cx(indices[6], indices[3])
    elif n==8:
        CGp(qc, indices[0], indices[1], 1/2)
        qc.cx(indices[1], indices[0])
        #
        CGp(qc, indices[0], indices[2], 1/2)
        qc.cx(indices[2], indices[0])
        #
        CGp(qc, indices[1], indices[3], 1/2)
        qc.cx(indices[3], indices[1])
        #
        CGp(qc, indices[0], indices[4], 1/2)
        qc.cx(indices[4], indices[0])
        #
        CGp(qc, indices[2], indices[5], 1/2)
        qc.cx(indices[5], indices[2])
        #
        CGp(qc, indices[1], indices[6], 1/2)
        qc.cx(indices[6], indices[1])
        #
        CGp(qc, indices[3], indices[7], 1/2)
        qc.cx(indices[7], indices[3])


def binstringToLabels_MaxKCut(k_cuts,num_V,binstring):
    k_bits = kBits_MaxKCut(k_cuts)
    label_list = [int(binstring[j*k_bits:(j+1)*k_bits], 2) for j in range(num_V)]
    label_string = ''
    for label in label_list:
        label_string += str(label)
    return label_string

def kBits_MaxKCut(k_cuts):
    return int(np.ceil(np.log2(k_cuts)))

def cost_MaxCut(labels, G, k_cuts):
    C = 0
    for edge in G.edges():
        i = int(edge[0])
        j = int(edge[1])
        li=min(k_cuts-1,int(labels[i]))## e.g. for k_cuts=3, labels 2 and 3 should be equal
        lj=min(k_cuts-1,int(labels[j]))## e.g. for k_cuts=3, labels 2 and 3 should be equal
        if li != lj:
            w = G[edge[0]][edge[1]]['weight']
            C += w
    return C



def validcoloring_onehot(s):
    num_ones=0
    for i in range(len(s)):
        num_ones+=int(s[i])
        if num_ones>1:
            break
    val = True
    if num_ones!=1:
        val = False
    return val

def validstring_onehot(s,num_V):
    if len(s)%num_V!=0:
        raise Exception("inconsistent lenght")
    l=int(len(s)/num_V)
    vale = True
    for i in range(num_V):
        ss=s[i*l:i*l+l]
        val=validcoloring_onehot(ss)
        #print(ss,val)
        if not val:
            break
    return val

def getcolor(s):
    for i in range(len(s)):
        if int(s[i])==1:
            return i
    return -1

def binstringToLabels_MaxKCut_onehot(labels, num_V, k_cuts):
    l=int(len(labels)/num_V)
    label_string=''
    for i in range(num_V):
        ss=labels[i*l:i*l+l]
        label_string+=str(getcolor(ss))
    return label_string

def measurementStatistics_MaxCut_onehot(experiment_results,  options=None):
    """
    Calculates the expectation and variance of the cost function. If
    results from multiple circuits are used as input, each circuit's
    expectation value is returned.
    :param experiment_results: Input on the form execute(...).result().results
    :param G: The graph on which the cost function is defined.
    :return: Lists of expectation values and variances
    """

    G = options.get('G', None)
    k_cuts = options.get('k_cuts', None)
    if G == None or k_cuts == False:
        raise Exception("Please specify options G and k_cuts")

    cost_best = -np.inf

    expectations = []
    variances = []
    num_V = G.number_of_nodes()
    for result in experiment_results:
        n_shots = result.shots
        counts = result.data.counts

        E = 0
        E2 = 0
        for hexkey in list(counts.keys()):
            count = counts[hexkey]
            binstring = "{0:b}".format(int(hexkey,0)).zfill(num_V*k_cuts)
            if validstring_onehot(binstring, num_V):
                labels = binstringToLabels_MaxKCut_onehot(binstring, num_V, k_cuts)
                cost = cost_MaxCut(labels,G, k_cuts)
                cost_best = max(cost_best, cost)
                E += cost*count;
                E2 += cost**2*count;

        if n_shots == 1:
            v = 0
        else:
            v = (E2-E**2)*n_shots/(n_shots-1)
        expectations.append(E/n_shots)
        variances.append(v)
    return expectations, variances, cost_best

def createCircuit_MaxCut(x, G, depth, k_cuts, version=1, usebarrier=False, name=None):

    num_V = G.number_of_nodes()
    k_bits = kBits_MaxKCut(k_cuts)
    if version==2:
        if k_cuts==2:
            Hij = np.array((-1, 1,
                             1,-1,))
        elif k_cuts==3:
            Hij = np.array((-1, 1, 1, 1,
                             1,-1, 1, 1,
                             1, 1,-1,-1,
                             1, 1,-1,-1))
        elif k_cuts==4:
            Hij = np.array((-1, 1, 1, 1,
                             1,-1, 1, 1,
                             1, 1,-1, 1,
                             1, 1, 1,-1))
        elif k_cuts==5:
            Hij = np.array((-1, 1, 1, 1, 1, 1, 1, 1, 
                             1,-1, 1, 1, 1, 1, 1, 1, 
                             1, 1,-1, 1, 1, 1, 1, 1, 
                             1, 1, 1,-1, 1, 1, 1, 1, 
                             1, 1, 1, 1,-1,-1,-1,-1, 
                             1, 1, 1, 1,-1,-1,-1,-1, 
                             1, 1, 1, 1,-1,-1,-1,-1, 
                             1, 1, 1, 1,-1,-1,-1,-1)) 
        elif k_cuts==6:
            Hij = np.array((-1, 1, 1, 1, 1, 1, 1, 1, 
                             1,-1, 1, 1, 1, 1, 1, 1, 
                             1, 1,-1, 1, 1, 1, 1, 1, 
                             1, 1, 1,-1, 1, 1, 1, 1, 
                             1, 1, 1, 1,-1, 1, 1, 1, 
                             1, 1, 1, 1, 1,-1,-1,-1, 
                             1, 1, 1, 1, 1,-1,-1,-1, 
                             1, 1, 1, 1, 1,-1,-1,-1)) 
        elif k_cuts==7:
            Hij = np.array((-1, 1, 1, 1, 1, 1, 1, 1, 
                             1,-1, 1, 1, 1, 1, 1, 1, 
                             1, 1,-1, 1, 1, 1, 1, 1, 
                             1, 1, 1,-1, 1, 1, 1, 1, 
                             1, 1, 1, 1,-1, 1, 1, 1, 
                             1, 1, 1, 1, 1,-1, 1, 1, 
                             1, 1, 1, 1, 1, 1,-1,-1, 
                             1, 1, 1, 1, 1, 1,-1,-1)) 
        elif k_cuts==8:
            Hij = np.array((-1, 1, 1, 1, 1, 1, 1, 1, 
                             1,-1, 1, 1, 1, 1, 1, 1, 
                             1, 1,-1, 1, 1, 1, 1, 1, 
                             1, 1, 1,-1, 1, 1, 1, 1, 
                             1, 1, 1, 1,-1, 1, 1, 1, 
                             1, 1, 1, 1, 1,-1, 1, 1, 
                             1, 1, 1, 1, 1, 1,-1, 1, 
                             1, 1, 1, 1, 1, 1, 1,-1)) 
        else:
            raise Exception("Circuit creation for k=",k_cuts," not implemented for version 2 (hard coded).")

    # we need 2 auxillary qubits if k is not a power of two
    num_aux=0
    k_is_power_of_two = math.log(k_cuts, 2).is_integer()
    if version==1 and not k_is_power_of_two:
        num_aux=2
        ind_a1=num_V * k_bits + num_aux - 2
        ind_a2=num_V * k_bits + num_aux - 1

    q = QuantumRegister(num_V * k_bits + num_aux)
    c = ClassicalRegister(num_V * k_bits)
    circ = QuantumCircuit(q, c, name=name)
    circ.h(range(num_V * k_bits))

    if usebarrier:
        circ.barrier()
    for d in range(depth):
        gamma = x[2 * d]
        beta = x[2 * d + 1]
        if version==2:
            for edge in G.edges():
                i = int(edge[0])
                j = int(edge[1])
                w = G[edge[0]][edge[1]]['weight']
                wg = w * gamma
                I = i * k_bits
                J = j * k_bits
                ind_Hij = [i_ for i_ in range(I, I+k_bits)]
                for j_ in range(J,J+k_bits):
                    ind_Hij.append(j_)
                U = np.diag(np.exp(-1j * (-wg) / 2 * Hij))
                circ.unitary(U, ind_Hij, 'Hij('+"{:.2f}".format(wg)+")")
        else:
            if k_cuts == 2:
                for edge in G.edges():
                    i = int(edge[0])
                    j = int(edge[1])
                    w = G[edge[0]][edge[1]]['weight']
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
            elif k_is_power_of_two:
                for edge in G.edges():
                    i = int(edge[0])
                    j = int(edge[1])
                    w = G[edge[0]][edge[1]]['weight']
                    wg = w * gamma
                    I = i * k_bits
                    J = j * k_bits
                    for k in range(k_bits):
                        circ.cx(I + k, J + k)
                        circ.x(J + k)
                    Cn_U3_0theta0(circ, [J-1+ind for ind in range(1,k_bits)], J+k_bits-1, -wg)
                    for k in reversed(range(k_bits)):
                        circ.x(J + k)
                        circ.cx(I + k, J + k)
                    if usebarrier:
                        circ.barrier()
            elif k_cuts == 3:
                for edge in G.edges():
                    i = int(edge[0])
                    j = int(edge[1])
                    w = G[edge[0]][edge[1]]['weight']
                    wg = w * gamma
                    I = i * k_bits
                    J = j * k_bits

                    for k in range(k_bits):
                        circ.cx(I + k, J + k)
                        circ.x(J + k)
                    Cn_U3_0theta0(circ, [J-1+ind for ind in range(1,k_bits)], J+k_bits-1, -wg)
                    for k in reversed(range(k_bits)):
                        circ.x(J + k)
                        circ.cx(I + k, J + k)
                    if usebarrier:
                        circ.barrier()
                    circ.x(I)
                    circ.ccx(I,I+1,ind_a1)
                    circ.ccx(J,J+1,ind_a2)
                    Cn_U3_0theta0(circ, [ind_a1, ind_a2], J+k_bits-1, -wg)
                    circ.ccx(J,J+1,ind_a2)
                    circ.ccx(I,I+1,ind_a1)
                    circ.x(I)
                    if usebarrier:
                        circ.barrier()
                    circ.x(J)
                    circ.ccx(I,I+1,ind_a1)
                    circ.ccx(J,J+1,ind_a2)
                    Cn_U3_0theta0(circ, [ind_a1, ind_a2], J+k_bits-1, -wg)
                    circ.ccx(J,J+1,ind_a2)
                    circ.ccx(I,I+1,ind_a1)
                    circ.x(J)

                    if usebarrier:
                        circ.barrier()
            elif k_cuts == 5:
                for edge in G.edges():
                    i = int(edge[0])
                    j = int(edge[1])
                    w = G[edge[0]][edge[1]]['weight']
                    wg = w * gamma
                    I = i * k_bits
                    J = j * k_bits

                    for k in range(k_bits):
                        circ.cx(I + k, J + k)
                        circ.x(J + k)
                    Cn_U3_0theta0(circ, [J-1+ind for ind in range(1,k_bits)], J+k_bits-1, -wg)
                    for k in reversed(range(k_bits)):
                        circ.x(J + k)
                        circ.cx(I + k, J + k)

                    if usebarrier:
                        circ.barrier()
                    circ.x(I)
                    circ.x(I+1)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    Cn_U3_0theta0(circ, [ind_a1, ind_a2], J+k_bits-1, -wg)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.x(I+1)
                    circ.x(I)
                    if usebarrier:
                        circ.barrier()
                    circ.x(J)
                    circ.x(J+1)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    Cn_U3_0theta0(circ, [ind_a1, ind_a2], J+k_bits-1, -wg)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.x(J+1)
                    circ.x(J)

                    if usebarrier:
                        circ.barrier()
                    circ.x(I+1)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    Cn_U3_0theta0(circ, [ind_a1, ind_a2], J+k_bits-1, -wg)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.x(I+1)
                    if usebarrier:
                        circ.barrier()
                    circ.x(J+1)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    Cn_U3_0theta0(circ, [ind_a1, ind_a2], J+k_bits-1, -wg)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.x(J+1)

                    if usebarrier:
                        circ.barrier()
                    circ.x(I)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    Cn_U3_0theta0(circ, [ind_a1, ind_a2], J+k_bits-1, -wg)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.x(I)
                    if usebarrier:
                        circ.barrier()
                    circ.x(J)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    Cn_U3_0theta0(circ, [ind_a1, ind_a2], J+k_bits-1, -wg)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.x(J)

                    if usebarrier:
                        circ.barrier()
            elif k_cuts == 6:
                for edge in G.edges():
                    i = int(edge[0])
                    j = int(edge[1])
                    w = G[edge[0]][edge[1]]['weight']
                    wg = w * gamma
                    I = i * k_bits
                    J = j * k_bits

                    for k in range(k_bits):
                        circ.cx(I + k, J + k)
                        circ.x(J + k)
                    Cn_U3_0theta0(circ, [J-1+ind for ind in range(1,k_bits)], J+k_bits-1, -wg)
                    for k in reversed(range(k_bits)):
                        circ.x(J + k)
                        circ.cx(I + k, J + k)

                    if usebarrier:
                        circ.barrier()
                    circ.x(I+1)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    Cn_U3_0theta0(circ, [ind_a1, ind_a2], J+k_bits-1, -wg)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.x(I+1)
                    if usebarrier:
                        circ.barrier()
                    circ.x(J+1)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    Cn_U3_0theta0(circ, [ind_a1, ind_a2], J+k_bits-1, -wg)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.x(J+1)

                    if usebarrier:
                        circ.barrier()
                    circ.x(I)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    Cn_U3_0theta0(circ, [ind_a1, ind_a2], J+k_bits-1, -wg)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.x(I)
                    if usebarrier:
                        circ.barrier()
                    circ.x(J)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    Cn_U3_0theta0(circ, [ind_a1, ind_a2], J+k_bits-1, -wg)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.x(J)

                    if usebarrier:
                        circ.barrier()
            elif k_cuts == 7:
                for edge in G.edges():
                    i = int(edge[0])
                    j = int(edge[1])
                    w = G[edge[0]][edge[1]]['weight']
                    wg = w * gamma
                    I = i * k_bits
                    J = j * k_bits

                    for k in range(k_bits):
                        circ.cx(I + k, J + k)
                        circ.x(J + k)
                    Cn_U3_0theta0(circ, [J-1+ind for ind in range(1,k_bits)], J+k_bits-1, -wg)
                    for k in reversed(range(k_bits)):
                        circ.x(J + k)
                        circ.cx(I + k, J + k)
                    if usebarrier:
                        circ.barrier()
                    circ.x(I)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    Cn_U3_0theta0(circ, [ind_a1, ind_a2], J+k_bits-1, -wg)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.x(I)
                    if usebarrier:
                        circ.barrier()
                    circ.x(J)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    Cn_U3_0theta0(circ, [ind_a1, ind_a2], J+k_bits-1, -wg)
                    circ.mcx([J,J+1,J+2],ind_a2)
                    circ.mcx([I,I+1,I+2],ind_a1)
                    circ.x(J)

                    if usebarrier:
                        circ.barrier()
            else:
                raise Exception("Circuit creation for k=",k_cuts," not implemented for version 1 (decomposed).")

        circ.rx(-2 * beta, range(num_V * k_bits))
        if usebarrier:
            circ.barrier()
    if version == 1 and not k_is_power_of_two:
        circ.measure(q[:-2], c)
    else:
        circ.measure(q, c)
    return circ

def createCircuit_MaxCut_onehot(x, G, depth, k_cuts, alpha=None, version=2, usebarrier=False, name=None):

    #W=np.zeros(2**k_cuts)
    #for i in range(k_cuts):
    #    bs='0b'
    #    for j in range(k_cuts):
    #        if i == j:
    #            bs+="1"
    #        else:
    #            bs+="0"
    #    ind=int(bs, 2)
    #    W[ind]=1
    #W = W/np.sqrt(k_cuts)

    num_V = G.number_of_nodes()

    num_qubits = num_V * k_cuts

    q = QuantumRegister(num_qubits)
    c = ClassicalRegister(num_qubits)
    circ = QuantumCircuit(q, c, name=name)
    if version==1:
        circ.h(range(num_qubits))
    else:
        for v in range(num_V):
            I = v*k_cuts
            Wn(circ, [i for i in range(I, I+k_cuts)])
            #circ.initialize(W, [q[i] for i in range(I, I+k_cuts)])

    if usebarrier:
        circ.barrier()
    for d in range(depth):
        gamma = x[2 * d]
        beta = x[2 * d + 1]
        # the objective Hamiltonian
        for edge in G.edges():
            i = int(edge[0])
            j = int(edge[1])
            w = G[edge[0]][edge[1]]['weight']
            wg = w * gamma
            I = k_cuts * i
            J = k_cuts * j
            for k in range(k_cuts):
                circ.cx(q[I+k], q[J+k])
                circ.rz(wg, q[J+k])
                circ.cx(q[I+k], q[J+k])
            if usebarrier:
                circ.barrier()
        # the penalty Hamiltonian
        if alpha != None:
            for v in range(num_V):
                I = v*k_cuts
                for i in range(k_cuts):
                    for j in range(i+1,k_cuts):
                        circ.cx(q[I+i], q[I+j])
                        circ.rz(gamma*alpha, q[I+j])
                        #circ.rz(alpha, q[I+j])
                        circ.cx(q[I+i], q[I+j])
                if usebarrier:
                    circ.barrier()
        if version==1:
            circ.rx(-2 * beta, range(num_qubits))
            if usebarrier:
                circ.barrier()
        else:
            for v in range(num_V):
                I = v*k_cuts
                ## odd
                for i in range(0,k_cuts-1,2):
                    circ.rxx(-2 * beta, q[I+i], q[I+i+1])
                    circ.ryy(-2 * beta, q[I+i], q[I+i+1])
                ## even
                for i in range(1,k_cuts,2):
                    circ.rxx(-2 * beta, q[I+i], q[I+(i+1)%k_cuts])
                    circ.ryy(-2 * beta, q[I+i], q[I+(i+1)%k_cuts])
                # final
                if k_cuts%2==1:
                    circ.rxx(-2 * beta, q[I+k_cuts-1], q[I])
                    circ.ryy(-2 * beta, q[I+k_cuts-1], q[I])
                if usebarrier:
                    circ.barrier()

    circ.measure(q, c)
    return circ


def find_max_cut_brute_force(G, k_cuts):
    if (len(G) > 30):
        raise Exception("Too many solutions to enumerate.")
    num_V = G.number_of_nodes()
    k_bits = kBits_MaxKCut(k_cuts)
    maxcut = []
    maxcut_value = 0
    for i in range((2*k_cuts) ** num_V):
        binstring = "{0:b}".format(i).zfill(num_V * k_bits)
        labels = binstringToLabels_MaxKCut(k_cuts, num_V, binstring)
        max_bin = int(max(labels))
        if max_bin>=k_cuts:
            continue
        cost = cost_MaxCut(labels, G, k_cuts)
        if (cost >= maxcut_value):
            maxcut = labels
            maxcut_value = cost
        if i % 1024 == 0:
            print(i / (2*k_cuts) ** num_V * 100, "%", end='\r')
    return maxcut_value, maxcut

def listSortedCosts_MaxCut(G, k_cuts):
    costs={}
    num_V = G.number_of_nodes()
    k_bits = kBits_MaxKCut(k_cuts)
    for i in range((2*k_cuts) ** num_V):
        binstring="{0:b}".format(i).zfill(k_bits * num_V)
        label_string = binstringToLabels_MaxKCut(k_cuts, num_V, binstring)
        max_bin = int(max(label_string))
        if max_bin>=k_cuts:
            continue
        costs[label_string]=cost_MaxCut(label_string, G, k_cuts)
        if i % 1024 == 0:
            print(i / (2*k_cuts) ** num_V * 100, "%", end='\r')
    sortedcosts={k: v for k, v in sorted(costs.items(), key=lambda item: item[1])}
    return sortedcosts

#def costsHist_MaxCut(G, k_cuts):
#    num_V = G.number_of_nodes()
#    costs=np.ones(k_cuts ** num_V)
#    k_bits = kBits_MaxKCut(k_cuts)
#    for i in range(k_cuts**num_V):
#        binstring="{0:b}".format(i).zfill(num_V * k_bits)
#        label_string = binstringToLabels_MaxKCut(k_cuts, num_V, binstring)
#        costs[i]= cost_MaxCut(label_string,G, k_cuts)
#    return costs

def costsHist_MaxCut(G, k_cuts):
    if k_cuts!=2:
        raise Exception("k_cuts must be equal to 2")
    num_V = G.number_of_nodes()
    costs=np.ones(2**num_V)
    for i in range(2**num_V):
        if i%1024*2*2*2==0:
            print(i/2**num_V*100, "%", end='\r')
        binstring="{0:b}".format(i).zfill(num_V)
        y=[int(i) for i in binstring]
        costs[i]=cost_MaxCut(y,G, k_cuts)
    print("100%")
    return costs


def bins_comp_basis(data, G, k_cuts):
    if k_cuts != 2:
        raise Exception("bins_comp_basis not implemented for k_cuts!=2")
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
        lc = cost_MaxCut(y,G, k_cuts)
        if lc==max_cost:
            max_solutions.append(y)
        elif lc>max_cost:
            max_solutions=[]
            max_solutions.append(y)
            max_cost=lc
        average_cost+=lc*counts
    return bins_states, max_cost, average_cost/num_shots, max_solutions


def measurementStatistics_MaxCut(experiment_results, options=None):
    """
    Calculates the expectation and variance of the cost function. If
    results from multiple circuits are used as input, each circuit's
    expectation value is returned.
    :param experiment_results: Input on the form execute(...).result().results
    :param G: The graph on which the cost function is defined.
    :return: Lists of expectation values and variances
    """

    G = options.get('G', None)
    k_cuts = options.get('k_cuts', None)
    if G == None or k_cuts == False:
        raise Exception("Please specify options G and k_cuts")

    k_bits = kBits_MaxKCut(k_cuts)
    cost_best = -np.inf

    expectations = []
    variances = []
    num_V = G.number_of_nodes()
    for result in experiment_results:
        n_shots = result.shots
        counts = result.data.counts

        E = 0
        E2 = 0
        for hexkey in list(counts.keys()):
            count = counts[hexkey]
            binstring = "{0:b}".format(int(hexkey,0)).zfill(num_V*k_bits)
            labels = binstringToLabels_MaxKCut(k_cuts,num_V,binstring)
            cost = cost_MaxCut(labels,G, k_cuts)
            cost_best = max(cost_best, cost)
            E += cost*count/n_shots;
            E2 += cost**2*count/n_shots;

        if n_shots == 1:
            v = 0
        else:
            v = (E2-E**2)*n_shots/(n_shots-1)
        expectations.append(E)
        variances.append(v)
    return expectations, variances, cost_best

def objective_function(params, G, backend, num_shots, k_cuts):
    """
    :return: minus the expectation value (in order to maximize MaxCut configuration)
    NB! If a list of circuits are run, only returns the expectation value of the first circuit.
    """
    qc = createCircuit_MaxCut(params, G, int(len(params)/2), k_cuts)
    if backend.configuration().local:
        job = execute(qc, backend, shots=num_shots)
    else:
        job = start_or_retrieve_job(name, backend, qc, options={'shots' : num_shots})
    res_data = job.results
    E,_,_ = measurementStatistics_MaxCut(res_data, options={'G' : G, 'k_cuts' : k_cuts})
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

# WARNING: While the following function does empirically seem to
# work, the theoretical backing should be double checked.
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


def optimize_random(K, G, backend, k_cuts, depth=1, decimals=0, num_shots=8192):
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
        sol = minimize(objective_function, x0=init_params, method='COBYLA', args=(G, backend, num_shots, k_cuts), constraints=cons)
        params = sol.x
        qc = createCircuit_MaxCut(params, G, depth, k_cuts)
        temp_res_data = execute(qc, backend, shots=num_shots).result().results
        [E],_,_ = measurementStatistics_MaxCut(temp_res_data, options={'G' : G, 'k_cuts' : k_cuts})
        avg_list[i] = E
        if E>record:
            record = E
            record_params = params
    return record_params, record, np.average(avg_list)

def scale_p(K, G, backend, k_cuts, depth=3, decimals=0, num_shots=8192):
    """
    :return: arrays of the p_values used, the corresponding array for the energy from the optimal
         energy config, and the average energy (for all the RIs at each p value)
    """
    H_list = np.zeros(depth)
    avg_list = np.zeros(depth)
    p_list = np.arange(1, depth + 1, 1)
    for d in range(1, depth + 1):
        temp, H_list[d-1], avg_list[d-1] = optimize_random(K, G, backend, k_cuts, d, decimals=decimals, num_shots=num_shots)
    return p_list, H_list, avg_list




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


def optimize_INTERP(K, G, backend, depth, k_cuts, decimals=0, num_shots=8192):
    """
    Optimizes the params using the INTERP heuristic
    :return: Array of the optimal parameters, and the correponding energy value
    """
    record = -np.inf
    for i in range(K):
        init_params = np.zeros(2)
        gamma_bounds, beta_bounds = parameterBounds_MaxCut(G, decimals=decimals)
        cons = COBYLAConstraints_MaxCut(gamma_bounds, beta_bounds, 1)
        sol = minimize(objective_function, x0=init_params, method='COBYLA', args=(G, backend, num_shots, k_cuts), constraints=cons)
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
            sol = minimize(objective_function, x0=init_params, method='COBYLA', args=(G, backend, num_shots, k_cuts), constraints=cons)
            params = sol.x
            init_gamma = params[0::2]
            init_beta = params[1::2]
        qc = createCircuit_MaxCut(params, G, depth, k_cuts)
        temp_res_data = execute(qc, backend, shots=num_shots).result().results
        [E],_,_ = measurementStatistics_MaxCut(temp_res_data, options={'G' : G, 'k_cuts' : k_cuts})
        if E>record:
            record = E
            record_params = params
    return record_params, record



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

global g_it
global g_values
global g_bestvalues
global g_gammabeta

def getval(gammabeta, backend, G, k_cuts, depth=1, version=version, noisemodel=None, shots=1024*2*2*2, name='', onehot=False, onehot_alpha=0):
    global g_it, g_values, g_bestvalues, g_gammabeta
    g_it+=1

    if onehot:
        circuit = createCircuit_MaxCut_onehot(gammabeta, G, depth, k_cuts, alpha=onehot_alpha, version=version, usebarrier=False, name=name)
    else:
        circuit = createCircuit_MaxCut(gammabeta, G, depth, k_cuts, version=version, usebarrier=False, name=name)
    if backend.configuration().local:
        job = execute(circuit, backend=backend, noise_model=noisemodel, shots=shots)
    else:
        job = start_or_retrieve_job(name+"_"+str(g_it), backend, circuit, options={'shots' : shots})

    if onehot:
        val,_,bval = measurementStatistics_MaxCut_onehot(job.result().results, options={'G' : G, 'k_cuts' : k_cuts})
    else:
        val,_,bval = measurementStatistics_MaxCut(job.result().results, options={'G' : G, 'k_cuts' : k_cuts})
    g_values[str(g_it)] = val[0]
    g_bestvalues[str(g_it)] = bval
    g_gammabeta[str(g_it)] = gammabeta
    return -val[0]

def runQAOA(G, k_cuts, backend, gamma_n, beta_n, gamma_max, beta_max, optmethod='COBYLA', circuit_version=1, shots=1024*2*2*2, name='', rerun=False, maxdepth=3, onehot=False, onehot_alpha=0):
    if k_cuts<2:
        raise Exception("k_cuts must be at least 2")
    repeats=5
    gammabetas = {}
    E = {}
    best = {}
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
                    if onehot:
                        circuits.append(createCircuit_MaxCut_onehot(np.array((gamma,beta)), G, depth, k_cuts, alpha=onehot_alpha, version=circuit_version, usebarrier=False, name=name+"_"+str(beta_n)+"_"+str(gamma_n)))
                    else:
                        circuits.append(createCircuit_MaxCut(np.array((gamma,beta)), G, depth, k_cuts, version=circuit_version, usebarrier=False, name=name+"_"+str(beta_n)+"_"+str(gamma_n)))
            job = execute(circuits, backend, shots=shots)
            if onehot:
                El,_,_ = measurementStatistics_MaxCut_onehot(job.result().results, options={'G' : G, 'k_cuts' : k_cuts})
            else:
                El,_,_ = measurementStatistics_MaxCut(job.result().results, options={'G' : G, 'k_cuts' : k_cuts})
            Elandscape = -np.array(El)
        else:
            Elandscape = np.zeros((beta_n, gamma_n))
            b=-1
            for beta in beta_grid:
                b+=1
                g=-1
                for gamma in gamma_grid:
                    g+=1
                    if onehot:
                        circuit = createCircuit_MaxCut_onehot(np.array((gamma,beta)), G, depth, k_cuts, alpha=onehot_alpha, version=circuit_version, usebarrier=False, name=name+"_"+str(b)+"_"+str(g))
                    else:
                        circuit = createCircuit_MaxCut(np.array((gamma,beta)), G, depth, k_cuts, version=circuit_version, usebarrier=False, name=name+"_"+str(b)+"_"+str(g))
                    job = start_or_retrieve_job(name+"_"+str(b)+"_"+str(g), backend, circuit, options={'shots' : shots})
                    #print("error message = ", job.error_message())
                    #job.error_message()
                    if onehot:
                        e,_,_ = measurementStatistics_MaxCut_onehot(job.result().results, options={'G' : G, 'k_cuts' : k_cuts})
                        print(e)
                    else:
                        e,_,_ = measurementStatistics_MaxCut(job.result().results, options={'G' : G, 'k_cuts' : k_cuts})
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
        out = minimize(getval, x0=x0, method=optmethod, args=(backend, G, k_cuts, depth, circuit_version, None, shots, name+"_opt_"+str(depth), onehot, onehot_alpha), options={'xatol': 1e-2, 'fatol': 1e-1, 'disp': True})#, constraints=cons)
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
            out = minimize(getval, x0=x0, method=optmethod, args=(backend, G, k_cuts, depth, circuit_version, None, shots, name+"_opt_"+str(depth), onehot, onehot_alpha), options={'xatol': 1e-2, 'fatol': 1e-1, 'disp': True})#, constraints=cons)
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
            out = minimize(getval, x0=x0, method=optmethod, args=(backend, G, k_cuts, depth, circuit_version, None, shots, name+"_opt_"+str(depth), onehot, onehot_alpha), options={'xatol': 1e-2, 'fatol': 1e-1, 'disp': True})#, constraints=cons)
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
            out = minimize(getval, x0=x0, method=optmethod, args=(backend, G, k_cuts, depth, circuit_version, None, shots, name+"_opt_"+str(depth), onehot, onehot_alpha), options={'xatol': 1e-2, 'fatol': 1e-1, 'disp': True})#, constraints=cons)
        ### pick the best value along the path
        ind = max(g_values, key=g_values.get)
        gammabetas['x0_d'+str(depth)] = x0.copy()
        gammabetas['xL_d'+str(depth)] = g_gammabeta[ind].copy()
        E[''+str(depth)] = g_values[ind]
        best[''+str(depth)] = g_bestvalues[ind]

    return Elandscape, gammabetas, E, best

def getStatistics(G, k_cuts, backend, gammabetas, circuit_version=1, shots=1024*2*2*2, maxdepth=3, name='', onehot=False, onehot_alpha=0):

    #num_shots = [i for i in range(2,2**6+1)]
    #num_shots.append(2**7)
    #num_shots.append(2**8)
    #num_shots.append(2**9)
    #num_shots.append(2**10)
    #num_shots.append(2**11)
    #num_shots.append(2**12)
    #num_shots.append(2**13)
    ##num_shots.append(2**14)
    num_shots=np.array([2**13,])

    av_max_cost = {}
    best_cost = {}
    distribution = {}

### ----------------------------
################
    depth=1
    print("depth =",depth)
################
### ----------------------------
    av_max_cost[str(depth)] = []
    best_cost[str(depth)] = []

    x = gammabetas['xL_d'+str(depth)]
    for ns in num_shots:
        if onehot:
            circ = createCircuit_MaxCut_onehot(x, G, depth, k_cuts, alpha=onehot_alpha, version=circuit_version, usebarrier=False, name=name+"d"+str(depth))
        else:
            circ = createCircuit_MaxCut(x, G, depth, k_cuts, version=circuit_version, usebarrier=False, name=name+"d"+str(depth))
        if backend.configuration().local:
            job = execute(circ, backend, shots=ns)
        else:
            job = start_or_retrieve_job(name+str(depth), backend, circ, options={'shots' : ns})
        if onehot:
            mc,_,bc = measurementStatistics_MaxCut_onehot(job.result().results, options={'G' : G, 'k_cuts' : k_cuts})
        else:
            mc,_,bc = measurementStatistics_MaxCut(job.result().results, options={'G' : G, 'k_cuts' : k_cuts})
        av_max_cost[str(depth)].append(mc)
        best_cost[str(depth)].append(bc)

    distribution[str(depth)] = job.result().get_counts(circ)

    if maxdepth>=2:
### ----------------------------
################
        depth=2
        print("depth =",depth)
################
### ----------------------------
        av_max_cost[str(depth)] = []
        best_cost[str(depth)] = []

        x = gammabetas['xL_d'+str(depth)]
        for ns in num_shots:
            if onehot:
                circ = createCircuit_MaxCut_onehot(x, G, depth, k_cuts, alpha=onehot_alpha, version=circuit_version, usebarrier=False, name=name+"d"+str(depth))
            else:
                circ = createCircuit_MaxCut(x, G, depth, k_cuts, version=circuit_version, usebarrier=False, name=name+"d"+str(depth))
            if backend.configuration().local:
                job = execute(circ, backend, shots=ns)
            else:
                job = start_or_retrieve_job(name+str(depth), backend, circ, options={'shots' : ns})
            if onehot:
                mc,_,bc = measurementStatistics_MaxCut_onehot(job.result().results, options={'G' : G, 'k_cuts' : k_cuts})
            else:
                mc,_,bc = measurementStatistics_MaxCut(job.result().results, options={'G' : G, 'k_cuts' : k_cuts})
            av_max_cost[str(depth)].append(mc)
            best_cost[str(depth)].append(bc)

        distribution[str(depth)] = job.result().get_counts(circ)

    if maxdepth>=3:
### ----------------------------
################
        depth=3
        print("depth =",depth)
################
### ----------------------------
        av_max_cost[str(depth)] = []
        best_cost[str(depth)] = []

        x = gammabetas['xL_d'+str(depth)]
        for ns in num_shots:
            if onehot:
                circ = createCircuit_MaxCut_onehot(x, G, depth, k_cuts, alpha=onehot_alpha, version=circuit_version, usebarrier=False, name=name+"d"+str(depth))
            else:
                circ = createCircuit_MaxCut(x, G, depth, k_cuts, version=circuit_version, usebarrier=False, name=name+"d"+str(depth))
            if backend.configuration().local:
                job = execute(circ, backend, shots=ns)
            else:
                job = start_or_retrieve_job(name+str(depth), backend, circ, options={'shots' : ns})
            if onehot:
                mc,_,bc = measurementStatistics_MaxCut_onehot(job.result().results, options={'G' : G, 'k_cuts' : k_cuts})
            else:
                mc,_,bc = measurementStatistics_MaxCut(job.result().results, options={'G' : G, 'k_cuts' : k_cuts})
            av_max_cost[str(depth)].append(mc)
            best_cost[str(depth)].append(bc)

        distribution[str(depth)] = job.result().get_counts(circ)

    if maxdepth>=4:
### ----------------------------
################
        depth=4
        print("depth =",depth)
################
### ----------------------------
        av_max_cost[str(depth)] = []
        best_cost[str(depth)] = []

        x = gammabetas['xL_d'+str(depth)]
        for ns in num_shots:
            if onehot:
                circ = createCircuit_MaxCut_onehot(x, G, depth, k_cuts, alpha=onehot_alpha, version=circuit_version, usebarrier=False, name=name+"d"+str(depth))
            else:
                circ = createCircuit_MaxCut(x, G, depth, k_cuts, version=circuit_version, usebarrier=False, name=name+"d"+str(depth))
            if backend.configuration().local:
                job = execute(circ, backend, shots=ns)
            else:
                job = start_or_retrieve_job(name+str(depth), backend, circ, options={'shots' : ns})
            if onehot:
                mc,_,bc = measurementStatistics_MaxCut_onehot(job.result().results, options={'G' : G, 'k_cuts' : k_cuts})
            else:
                mc,_,bc = measurementStatistics_MaxCut(job.result().results, options={'G' : G, 'k_cuts' : k_cuts})
            av_max_cost[str(depth)].append(mc)
            best_cost[str(depth)].append(bc)

        distribution[str(depth)] = job.result().get_counts(circ)

    return num_shots, av_max_cost, best_cost, distribution

