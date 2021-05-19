from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
import numpy as np

import sys
sys.path.append('../')

from qiskit_utilities.utilities import *

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

def createCircuit_MaxCut(x, depth, options=None):

    G = options.get('G', None)
    k_cuts = options.get('k_cuts', None)
    version = options.get('version', 1)
    usebarrier = options.get('usebarrier', False)
    name = options.get('name', None)

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

def createCircuit_MaxCut_onehot(x, depth, options=None):

    G = options.get('G', None)
    k_cuts = options.get('k_cuts', None)
    alpha = options.get('alpha', None)
    version = options.get('version', 2)
    usebarrier = options.get('usebarrier', False)
    name = options.get('name', None)

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
