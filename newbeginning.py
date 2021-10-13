import numpy as np
import itertools

from qiskit import *
from scipy.optimize import minimize

class QAOAbase:

    def __init__(self, CR, FR):
        self.CR=CR
        self.FR=FR
        self.F, self.R  = np.shape(self.FR)

    def createCircuit(self):
        raise NotImplementedError

    def cost2(self, binstring,plotsolutions=False):
        x = np.array(list(map(int,binstring)))
        exco=np.sum((1 - (self.FR @ x))**2)
        if self.CR is None:
            cost=0.
        else:
            cost=(self.CR @ x)
        if plotsolutions and exco==0:
            print(x, (self.CR @ x))
        return  cost, exco

    def cost(self, binstring,plotsolutions=False):
        x = np.array(list(map(int,binstring)))
        exco=np.sum((np.sum(self.FR*x,1) -1)**2)
        if self.CR is None:
            cost=0.
        else:
            cost=np.sum(x*(self.CR**2))
        return  cost, exco

    def cost_vector(self, mu, plotsolutions=False):
        state_strings = np.array([''.join(i) for i in itertools.product('01', repeat= self.R)])
        costs=np.zeros((2**self.R))
        co=np.zeros((2**self.R))
        ex=np.zeros((2**self.R))
        for i in range(2**self.R):
            # Reverse string since qiskit uses ordering MSB ... LSB
            co[i], ex[i] = self.cost(state_strings[i][::-1], plotsolutions=plotsolutions)
            costs[i] = co[i]+mu*ex[i]
        if plotsolutions:
            print("min cost=", np.min(costs))
        return costs, co, ex


    def measurementStatistics(self, job, nb=None, ng=None, nd=None, mu=1, usestatevec=True):
        if usestatevec:
            costs, co, ex =self.cost_vector(mu)
        if nb==None and ng==None and nd==None:
            E=0
            Ecost=0
            Eexco=0
            if usestatevec:
                statevector = job.result().results[0].data.statevector
                probs = np.abs(statevector)**2
                E = costs @ probs
                Ecost = co @ probs
                Eexco = ex @ probs
            else:
                res_dict = job.result().get_counts()
                for key in res_dict:
                    # Reverse string since qiskit uses ordering MSB ... LSB
                    co, ex = self.cost(key[::-1])
                    E += res_dict[key]*(co+mu*ex)
                    Ecost += res_dict[key]*co
                    Eexco += res_dict[key]*mu
        elif self.num_params==2:
            E=np.zeros((nb,ng))
            Ecost=np.zeros((nb,ng))
            Eexco=np.zeros((nb,ng))
            for i in range(nb):
                for j in range(ng):
                    if usestatevec:
                        statevector = job.result().results[j+ng*i].data.statevector
                        probs = np.abs(statevector)**2
                        E[i,j] = costs @ probs
                        Ecost[i,j] = co @ probs
                        Eexco[i,j] = ex @ probs
                    else:
                        res_dict = job.result().get_counts()[j+ng*i]
                        for key in res_dict:
                            # Reverse string since qiskit uses ordering MSB ... LSB
                            co, ex = self.cost(key[::-1])
                            E[i,j] += res_dict[key]*(co+mu*ex)
                            Ecost[i,j] += res_dict[key]*co
                            Eexco[i,j] += res_dict[key]*ex
        else:
            E=np.zeros((nb,ng,nd))
            Ecost=np.zeros((nb,ng,nd))
            Eexco=np.zeros((nb,ng,nd))
            for i in range(nb):
                for j in range(ng):
                    for l in range(nd):
                        statevector = job.result().results[l+nd*(j+ng*i)].data.statevector
                        probs = np.abs(statevector)**2
                        E[i,j,l] = costs @ probs
                        Ecost[i,j,l] = co @ probs
                        Eexco[i,j,l] = ex @ probs
        return E, Ecost, Eexco


    def mix_statesX(self, qc, beta):
        qc.rx( - 2*beta, range(qc.num_qubits))
        return qc

    def mix_statesY(self, qc, beta):
        qc.ry( - 2*beta, range(qc.num_qubits))
        return qc

    def mix_states(self, qc, beta, binstring):
        if binstring=='01<->10_ind01':
            qc.rxx(-beta, 0, 1)
            qc.ryy(-beta, 0, 1)
            qc.rxx(-beta, 1, 0)
            qc.ryy(-beta, 1, 0)
        if binstring=='01<->10_ind12':
            qc.rxx(-beta, 1, 2)
            qc.ryy(-beta, 1, 2)
            qc.rxx(-beta, 2, 1)
            qc.ryy(-beta, 2, 1)
        elif binstring=='001<->110':
            #A=np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
            #            [0., 0., 0., 0., 0., 0., 1., 0.],
            #            [0., 0., 0., 0., 0., 0., 0., 0.],
            #            [0., 0., 0., 0., 0., 0., 0., 0.],
            #            [0., 0., 0., 0., 0., 0., 0., 0.],
            #            [0., 0., 0., 0., 0., 0., 0., 0.],
            #            [0., 1., 0., 0., 0., 0., 0., 0.],
            #            [0., 0., 0., 0., 0., 0., 0., 0.]])

            #U = np.exp(-1j*beta*A)
            #print(U)
            #qc.unitary(U, range(3), 'A('+"{:.2f}".format(beta)+")")
            qc.barrier()
            # XXX
            for j in range(3):
                qc.h(j)
            qc.cx(2,1)
            qc.cx(1,0)
            qc.rz(-beta/2, 0)
            qc.cx(1,0)
            qc.cx(2,1)
            for j in range(3):
                qc.h(j)

            qc.barrier()
            # XYY
            qc.s(0)
            qc.s(1)
            for j in range(3):
                qc.h(j)
            qc.cx(2,1)
            qc.cx(1,0)
            qc.rz(-beta/2, 0)
            qc.cx(1,0)
            qc.cx(2,1)
            for j in range(3):
                qc.h(j)
            qc.sdg(0)
            qc.sdg(1)

            qc.barrier()
            # YXY
            qc.s(0)
            qc.s(2)
            for j in range(3):
                qc.h(j)
            qc.cx(2,1)
            qc.cx(1,0)
            qc.rz(-beta/2, 0)
            qc.cx(1,0)
            qc.cx(2,1)
            for j in range(3):
                qc.h(j)
            qc.sdg(0)
            qc.sdg(2)

            qc.barrier()
            # -YYX
            qc.s(1)
            qc.s(2)
            for j in range(3):
                qc.h(j)
            qc.cx(2,1)
            qc.cx(1,0)
            qc.rz(+beta/2, 0)
            qc.cx(1,0)
            qc.cx(2,1)
            for j in range(3):
                qc.h(j)
            qc.sdg(1)
            qc.sdg(2)
            qc.barrier()
        else:
            raise NotImplementedError
        return qc


    def apply_exco2(self, qc, gamma):
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


    def apply_exco(self, qc, gamma):
        for i in range(qc.num_qubits):
            w=0
            for j in range(self.F):
                w += .5 * self.FR[j,i]*(np.sum(self.FR[j,:])-2)
            if self.CR is not None:
                w += .25 * self.CR[i]**2
            if abs(w)>1e-14:
                wg = w * gamma
                qc.rz(wg, i)
            ###
            for j in range(i+1, self.R):
                w=0
                for k in range(self.F):
                    w += 0.5 * self.FR[k,i]*self.FR[k,j]
                    
                if w>0:
                    wg = w * gamma
                    qc.cx(i, j)
                    qc.rz(wg, j)
                    qc.cx(i, j)
        return qc

    def apply_cost(self, qc, gamma):
        for r in range(qc.num_qubits):
            hr = 0.5 * self.CR[r]
            if not np.isclose(hr, 0):
                qc.rz( 2*gamma * hr, r)
        return qc

    def getElandscape(self, backend, mu,useExco=None, gamma_max=2*np.pi,beta_max=np.pi,delta_max=2*np.pi,ng=40,nb=20,nd=40, barrier=False, sv=None, mixerbinstring=None):
        depth=1
        circuits=[]
        if self.num_params==2:
            for beta in np.linspace(0,beta_max,nb,endpoint=False):
                for gamma in np.linspace(0,gamma_max,ng,endpoint=False):

                    if useExco is not None:
                        qc=self.createCircuit(np.array((gamma,beta)), useExco, barrier=barrier, sv=sv, mixerbinstring=mixerbinstring)
                    else:
                        qc=self.createCircuit(np.array((gamma,beta)), mu, depth, barrier=barrier, sv=sv, mixerbinstring=mixerbinstring)

                    circuits.append(qc)
        else:
            for delta in np.linspace(0,delta_max,nd,endpoint=False):
                for beta in np.linspace(0,beta_max,nb,endpoint=False):
                    for gamma in np.linspace(0,gamma_max,ng,endpoint=False):
                        qc=self.createCircuit(np.array((gamma,beta,delta)), mu, depth, barrier=barrier, sv=sv, mixerbinstring=mixerbinstring)

                        circuits.append(qc)

        job = execute(circuits, backend)
        E, Ecost, Eexco = self.measurementStatistics(job, nb, ng, nd, mu=mu)

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

        return E, Ecost, Eexco, x0, job, index

    global g_it
    global g_jobs
    global g_values
    global g_x

    def getval(self, x, backend, mu, useExco, depth, sv, mixerbinstring):
        global g_it, g_jobs, g_values, g_x
        g_it+=1

        if useExco is not None:
            qc = self.createCircuit(x, useExco, sv=sv, mixerbinstring=mixerbinstring)
        else:
            qc = self.createCircuit(x, mu, depth, sv=sv, mixerbinstring=mixerbinstring)

        job = execute(qc, backend)

        val, _, _ = self.measurementStatistics(job, mu=mu)

        g_values[str(g_it)] = val
        g_jobs[str(g_it)] = job
        g_x[str(g_it)] = x
        return val

    def getlocalmin(self, x0, backend, mu, useExco=None, depth=1, barrier=False, sv=None, mixerbinstring=None, method="Nelder-Mead"):

        global g_it, g_jobs, g_values, g_x
        g_it=0
        g_jobs={}
        g_values={}
        g_x={}

        out = minimize(self.getval, x0=x0, method=method, args=(backend, mu, useExco, depth, sv, mixerbinstring), options={'xatol': 1e-2, 'fatol': 1e-1, 'disp': True})#, constraints=cons)
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
                ,extent=[gamma[0]-shiftg, gamma[-1]+shiftg, beta[0]-shiftb, beta[-1]+shiftb], aspect=1)
    pl.xlabel('$\gamma$',loc='left')
    pl.ylabel(r'$\beta$')
    pl.colorbar(shrink=0.75, pad=0.05, orientation="horizontal")
    pl.xticks(np.linspace(0, gamma_max,10,endpoint=False))#, ['0', r'$\pi$', r'$2\pi$'])
    pl.yticks(np.linspace(0, beta_max,5,endpoint=False))#, ['0', r'$\pi$', r'$2\pi$'])
    #pl.yticks([0,beta_max], ['0', r'$\pi/2$'])
    pl.xlabel('$\gamma$',loc='left')
    pl.ylabel(r'$\beta$')
    return fig


def getSpectrum(CR, FR, mumax=2):
    Ham=False
    if Ham:
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
    else:

        qaoa = QAOAbase(CR, FR)
        n=2
        R=CR.shape[0]
        ue = np.zeros((2**R,n))
        x=np.linspace(0,mumax,n)

        ue={}
        lab={}
        for i in range(2**R):
            binstr="{0:b}".format(i).zfill(R)
            lab[i]=binstr
            co, ex = qaoa.cost(binstr,0)
            ue[i]=np.array((co, co+mumax*ex))
        return x, ue,ue,ue,lab


class QAOAChoose(QAOAbase):

    num_params=2
    #def __init__(self, CR, FR):
    #    super().__init__(CR, FR)

    def createCircuit(self, x,useExco,barrier=False,sv=None, mixerbinstring=None):
        gamma=x[::2]
        beta=x[1::2]
        qc = QuantumCircuit(self.R)
        if sv is not None:
            qc.initialize(sv)
        else:
            qc.h(range(self.R))
        if barrier:
            qc.barrier()
        i=-1
        for ue in useExco:
            i+=1
            if ue:
                qc = self.apply_exco(qc, gamma[i])
                qc = self.mix_statesX(qc, beta[i])
            else:
                qc = self.apply_cost(qc, gamma[i])
                if mixerbinstring is None:
                    qc = self.mix_statesX(qc, beta[i])
                #else:
                    #qc = self.mix_states(qc, beta[i], mixerbinstring)
                    #qc = self.mix_states(qc, beta[i], '01<->10_ind12')
                    qc.rx( - 2*beta[i], 2)

            if barrier:
                qc.barrier()
            if barrier:
                qc.barrier()
                qc.barrier()
        return qc

class QAOANor(QAOAbase):

    num_params=3

    def createCircuit(self, x,mu,depth,barrier=False,sv=None, mixerbinstring=None):
        gamma=x[::3]
        beta=x[1::3]
        delta=x[2::3]
        qc = QuantumCircuit(self.R)
        if sv is not None:
            qc.initialize(sv)
        else:
            qc.h(range(self.R))
        if barrier:
            qc.barrier()
        i=-1
        for d in range(depth):
            i+=1
            qc = self.apply_exco(qc, gamma[i])
            qc = self.mix_statesX(qc, beta[i])
            qc = self.apply_cost(qc, delta[i])
            qc = self.mix_statesX(qc, beta[i])
            if barrier:
                qc.barrier()
                qc.barrier()
        return qc

class QAOASwe(QAOAbase):

    num_params=2

    def createCircuit(self, x,mu,depth,barrier=False,sv=None, mixerbinstring=None):
        gamma=x[::2]
        beta=x[1::2]
        qc = QuantumCircuit(self.R)
        if sv is not None:
            qc.initialize(sv)
        else:
            qc.h(range(self.R))
        if barrier:
            qc.barrier()
        i=-1
        for d in range(depth):
            i+=1
            qc = self.apply_exco(qc, mu*gamma[i])
            qc = self.apply_cost(qc, gamma[i])
            if barrier:
                qc.barrier()
            qc = self.mix_statesX(qc, beta[i])
            if barrier:
                qc.barrier()
                qc.barrier()
        return qc


