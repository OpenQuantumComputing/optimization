import numpy as np
import itertools

from qiskit import *
from scipy.optimize import minimize

class QAOAbase:

    def __init__(self, CR, FR):
        self.CR=CR
        self.FR=FR

    def createCircuit(self):
        raise NotImplementedError

    def cost(self, binstring,mu=1, plotsolutions=False):
        # Reverse string since qiskit uses ordering MSB ... LSB
        x = np.array(list(map(int,binstring[::-1])))
        if self.CR is None:
            return  np.sum((1 - (self.FR @ x))**2)
        else:
            excost=np.sum((1 - (self.FR @ x))**2)
            if plotsolutions and excost==0:
                print(x, (self.CR @ x))
            return  ( (self.CR @ x) + mu *excost  )

    def measurementStatisticsDebug(self, job, nb, ng, nd=None, mu=1, usestatevec=True):
        costs=self.cost_vector(mu)
        if self.num_params==2:
            E=np.zeros((nb,ng))
            for i in range(nb):
                for j in range(ng):
                    statevector = job.result().results[j+ng*i].data.statevector
                    probs = np.abs(statevector)**2
                    res_dict = job.result().get_counts()
                    print(probs)
                    print(costs)
                    print(res_dict)
                    e1=0
                    for key in res_dict:
                        e1 += res_dict[key]*self.cost(key, mu=mu)
                    e2=costs @ probs
                    print("E=",e1,e2)
        else:
            E=np.zeros((nb,ng,nd))
            for i in range(nb):
                for j in range(ng):
                    for l in range(nd):
                        statevector = job.result().results[l+nd*(j+ng*i)].data.statevector
                        probs = np.abs(statevector)**2
                        E[i,j,l] = costs @ probs[::-1]
    #             print(costs,probs)
        return E


    def measurementStatistics(self, job, nb=None, ng=None, nd=None, mu=1, usestatevec=True):
        if usestatevec:
            costs=self.cost_vector(mu)
        if nb==None and ng==None and nd==None:
            E=0
            if usestatevec:
                statevector = job.result().results[0].data.statevector
                probs = np.abs(statevector)**2
                E = costs @ probs
            else:
                res_dict = job.result().get_counts()
                for key in res_dict:
                    E += res_dict[key]*self.cost(key, mu=mu)
        elif self.num_params==2:
            E=np.zeros((nb,ng))
            for i in range(nb):
                for j in range(ng):
                    if usestatevec:
                        statevector = job.result().results[j+ng*i].data.statevector
                        probs = np.abs(statevector)**2
                        E[i,j] = costs @ probs
                    else:
                        res_dict = job.result().get_counts()[j+ng*i]
                        for key in res_dict:
                            E[i,j] += res_dict[key]*self.cost(key, mu=mu)
        else:
            E=np.zeros((nb,ng,nd))
            for i in range(nb):
                for j in range(ng):
                    for l in range(nd):
                        statevector = job.result().results[l+nd*(j+ng*i)].data.statevector
                        probs = np.abs(statevector)**2
                        E[i,j,l] = costs @ probs
    #             print(costs,probs)
        return E


    def cost_vector(self, mu, plotsolutions=False):
        F, R  = np.shape(self.FR)
        state_strings = np.array([''.join(i) for i in itertools.product('01', repeat= R)])
        costs=np.zeros((2**R))
        for i in range(2**R):
            costs[i] = self.cost(state_strings[i], mu=mu, plotsolutions=plotsolutions)
        #     costs[i] = cost(state_strings[2**R-i-1],self.CR=self.CR, FR=FR, mu=1)
        if plotsolutions:
            print("min cost=", np.min(costs))
        return costs

    def mix_states(self, qc, beta):
        qc.rx( - 2*beta, range(qc.num_qubits))
        return qc

    def apply_exco(self, qc, FR, gamma):
        for r in range(qc.num_qubits):

            hr = 0.5 * self.FR[:,r] @ (np.sum(self.FR,axis = 1) - 2)
            if not np.isclose(hr, 0):
                qc.rz( 2*gamma * hr, range(qc.num_qubits))

            for r_ in range(r+1,qc.num_qubits):
                Jrr_  = 0.5 * self.FR[:,r] @ self.FR[:,r_]
                if not np.isclose(Jrr_, 0):
                    qc.cx(r, r_)
                    qc.rz(2*gamma * Jrr_, r_)
                    qc.cx(r, r_)
        return qc

    def apply_cost(self, qc, CR, gamma):
        for r in range(qc.num_qubits):
            hr = 0.5 * self.CR[r]
            if not np.isclose(hr, 0):
                qc.rz( 2*gamma * hr, r)
        return qc

    def getElandscape(self, backend, mu,useExco=None, gamma_max=2*np.pi,beta_max=np.pi,delta_max=2*np.pi,ng=40,nb=20,nd=40, barrier=False, sv=None):
        depth=1
        circuits=[]
        if self.num_params==2:
            for beta in np.linspace(0,beta_max,nb,endpoint=False):
                for gamma in np.linspace(0,gamma_max,ng,endpoint=False):

                    if useExco is not None:
                        qc=self.createCircuit(np.array((gamma,beta)), useExco, barrier=barrier, sv=sv)
                    else:
                        qc=self.createCircuit(np.array((gamma,beta)), mu, depth, barrier=barrier, sv=sv)

                    circuits.append(qc)
        else:
            for delta in np.linspace(0,delta_max,nd,endpoint=False):
                for beta in np.linspace(0,beta_max,nb,endpoint=False):
                    for gamma in np.linspace(0,gamma_max,ng,endpoint=False):
                        qc=self.createCircuit(np.array((gamma,beta,delta)), mu, depth, barrier=barrier, sv=sv)

                        circuits.append(qc)

        job = execute(circuits, backend)
        E = self.measurementStatistics(job, nb, ng, nd, mu=mu)

        if self.num_params==2:
            i_b,i_g=np.where(E==np.min(E))
            i_b=i_b[0]
            i_g=i_g[0]
            gamma=np.linspace(0,gamma_max,ng,endpoint=False)
            beta=np.linspace(0,beta_max,nb,endpoint=False)
            x0=np.array((gamma[i_g],beta[i_b]))
            index=i_g+ng*i_b
        else:
            i_b,i_g,i_d=np.where(E==np.min(E))
            i_b=i_b[0]
            i_g=i_g[0]
            i_d=i_d[0]
            gamma=np.linspace(0,gamma_max,ng,endpoint=False)
            beta=np.linspace(0,beta_max,nb,endpoint=False)
            delta=np.linspace(0,delta_max,nd,endpoint=False)
            x0=np.array((gamma[i_g],beta[i_b],delta[i_d]))
            index=i_d+nd*(i_g+ng*i_b)

        return E, x0, job, index

    global g_it
    global g_jobs
    global g_values
    global g_x

    def getval(self, x, backend, mu, useExco, depth, sv):
        global g_it, g_jobs, g_values, g_x
        g_it+=1

        if useExco is not None:
            qc = self.createCircuit(x, useExco, sv=sv)
        else:
            qc = self.createCircuit(x, mu, depth, sv=sv)

        job = execute(qc, backend)

        val = self.measurementStatistics(job, mu=mu)

        g_values[str(g_it)] = val
        g_jobs[str(g_it)] = job
        g_x[str(g_it)] = x
        return val

    def getlocalmin(self, x0, backend, mu, useExco=None, depth=1, barrier=False, sv=None, method="Nelder-Mead"):

        global g_it, g_jobs, g_values, g_x
        g_it=0
        g_jobs={}
        g_values={}
        g_x={}

        out = minimize(self.getval, x0=x0, method=method, args=(backend, mu, useExco, depth, sv), options={'xatol': 1e-2, 'fatol': 1e-1, 'disp': True})#, constraints=cons)
        ind = min(g_values, key=g_values.get)
        return out, g_jobs[ind], g_x[ind]


import matplotlib.pyplot as pl

def getfig(E, beta_max=np.pi,gamma_max=2*np.pi,nb=20,ng=40):
    gamma=np.linspace(0,gamma_max,ng,endpoint=False)
    beta=np.linspace(0,beta_max,nb,endpoint=False)
    shiftg=0.5*(gamma[1]-gamma[0])
    shiftb=0.5*(beta[1]-beta[0])

    fig=pl.figure(figsize=(20,10), dpi=200, facecolor='w', edgecolor='k');
    pl.imshow(E,interpolation='nearest',origin='lower'
                ,extent=[shiftg,gamma_max-shiftg,shiftb,beta_max-shiftb], aspect=1)
    pl.xlabel('$\gamma$',loc='left')
    pl.ylabel(r'$\beta$')
    pl.colorbar(shrink=0.75, pad=0.05, orientation="horizontal")
    pl.xticks(np.linspace(0, gamma_max,9,endpoint=False))#, ['0', r'$\pi$', r'$2\pi$'])
    pl.yticks(np.linspace(0, beta_max,5,endpoint=False))#, ['0', r'$\pi$', r'$2\pi$'])
    #pl.yticks([0,beta_max], ['0', r'$\pi/2$'])
    pl.xlabel('$\gamma$',loc='left')
    pl.ylabel(r'$\beta$')
    return fig


def getSpectrum(CR, FR):
    I=np.array((1,1))
    Z=np.array((1,-1))

    R=CR.shape[0]

    H_cost = np.zeros(2**R)
    H_exco = np.zeros(2**R)
    for r in range(R):
        for i in range(R):
            if i==0:
                if i==r:
                    Zr = Z
                else:
                    Zr = I
            else:
                if i==r:
                    Zr = np.kron(Z,Zr)
                else:
                    Zr = np.kron(I,Zr)
        H_cost += 0.5*CR[r]*Zr
        H_exco += 0.5 * FR[:,r] @ (np.sum(FR,axis = 1) - 2)*Zr

    for r in range(R):
        for r_ in range(r+1,R):
            for i in range(R):
                if i==0:
                    if i==r or i==r_:
                        ZZrr_ = Z
                    else:
                        ZZrr_ = I
                else:
                    if i==r or i==r_:
                        ZZrr_ = np.kron(Z,ZZrr_)
                    else:
                        ZZrr_ = np.kron(I,ZZrr_)
            Jrr_  = 0.5 * FR[:,r] @ FR[:,r_]
            H_exco+=Jrr_*ZZrr_

    n=100
    e = np.zeros((2**R,n))
    x=np.linspace(0,4,n)
    # E=[]
    for i in range(n):
        e[:,i]=-H_cost+x[i]*H_exco
    #     if i==0:
    #     for j in range(len(E)):
    #         found=False
    #         if abs(E[j]-e[j,])<1e-8
    #     e[:,i] /=np.max(np.abs(e[:,i]))

    emax=np.max(e,axis=0)
    emin=np.min(e,axis=0)

    es = np.zeros((2**R,n))
    for i in range(2**R):
        es[i,:]=(e[i,:]-emin)/(emax-emin)

    ue={}
    lab={}
    j=0
    ci=33
    for i in range(2**R):
        used=False
        ik=None
    #     print()
    #     print("new i",i)
    #     print(ue)
        for key in ue:
    #     for j in range(len(ue)):
    #         print("key=", key, e[i,0], ue[key][0])
    #         print(ue)
            if abs(e[i,ci]-ue[key][ci])<1e-8:
                used=True
                ik=key
                break
    #     print("used=", used)
        if used:
            print(e[i,ci], "already used")
            lab[ik]+=", "+"{0:b}".format(i).zfill(R)
        else:
            j+=1
            ue[j] = e[i,:]
            lab[j]="{0:b}".format(i).zfill(R)
    # for j in range(len(ue)):
    #         pl.plot(x,ue[i,:], label="{0:b}".format(i).zfill(3))
    return x, e, es, ue, lab

class QAOAChoose(QAOAbase):

    num_params=2
    #def __init__(self, CR, FR):
    #    super().__init__(CR, FR)

    def createCircuit(self, x,useExco,barrier=False,sv=None):
        F, R  = np.shape(self.FR)
        gamma=x[::2]
        beta=x[1::2]
        qc = QuantumCircuit(R)
        if sv is not None:
            qc.initialize(sv)
            if barrier:
                qc.barrier()
        qc.h(range(R))
        if barrier:
            qc.barrier()
        i=-1
        for ue in useExco:
            i+=1
            if ue:
                qc = self.apply_exco(qc, self.FR, gamma[i])
            else:
                qc = self.apply_cost(qc, self.CR, gamma[i])
            if barrier:
                qc.barrier()
            qc = self.mix_states(qc, beta[i])
            if barrier:
                qc.barrier()
                qc.barrier()
        return qc

class QAOANor(QAOAbase):

    num_params=3

    def createCircuit(self, x,mu,depth,barrier=False,sv=None):
        F, R  = np.shape(self.FR)
        gamma=x[::3]
        beta=x[1::3]
        delta=x[2::3]
        qc = QuantumCircuit(R)
        if sv is not None:
            qc.initialize(sv)
            if barrier:
                qc.barrier()
        qc.h(range(R))
        if barrier:
            qc.barrier()
        i=-1
        for d in range(depth):
            i+=1
            qc = self.apply_exco(qc, self.FR, gamma[i])
            qc = self.mix_states(qc, beta[i])
            qc = self.apply_cost(qc, self.CR, delta[i])
            qc = self.mix_states(qc, beta[i])
            if barrier:
                qc.barrier()
                qc.barrier()
        return qc

class QAOASwe(QAOAbase):

    num_params=2

    def createCircuit(self, x,mu,depth,barrier=False,sv=None):
        F, R  = np.shape(self.FR)
        gamma=x[::2]
        beta=x[1::2]
        qc = QuantumCircuit(R)
        if sv is not None:
            qc.initialize(sv)
            if barrier:
                qc.barrier()
        qc.h(range(R))
        if barrier:
            qc.barrier()
        i=-1
        for d in range(depth):
            i+=1
            qc = self.apply_exco(qc, self.FR, mu*gamma[i])
            qc = self.apply_cost(qc, self.CR, gamma[i])
            if barrier:
                qc.barrier()
            qc = self.mix_states(qc, beta[i])
            if barrier:
                qc.barrier()
                qc.barrier()
        return qc


