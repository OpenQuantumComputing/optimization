from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

fontsize = 24
newparams = {'axes.titlesize': fontsize,
             'axes.labelsize': fontsize,
             'ytick.labelsize': fontsize,
             'xtick.labelsize': fontsize, 
             'legend.fontsize': fontsize,
             'figure.titlesize': fontsize,
             'legend.handlelength': 1.5, 
             'lines.linewidth': 2,
             'lines.markersize': 7,
             'figure.figsize': (12, 7), 
             'figure.dpi':200,
             'text.usetex' : True,
             'font.family' : "sans-serif"
            }

plt.rcParams.update(newparams)

def plot_H_prob(qaoa, SP, C, savefig = None):

    best_cost = np.max( qaoa.vector_cost(qaoa.state_strings) )

    fig, ax = plt.subplots(figsize  = (14,7))

    ax.set_ylabel(r"$\overline{\langle H_{\mathrm{tot}} \rangle}/\min \limits_{\mathbf{x}} \langle H_{\mathrm{tot}} \rangle $", color = "black")
    ax.hlines(1,1,qaoa.max_depth,color = "black", ls = "--")
    ax.plot(np.arange(1,qaoa.max_depth +1, 1), C/best_cost, color = "black", ls = "-", marker = "o")

    ax.grid(ls = "--")
    ax.set_xlabel("Depth, $d$")
    #ax.set_yscale("log")

    ax2 = plt.twinx()

    ax2.plot(np.arange(1,qaoa.max_depth+1,1), SP, color ="blue", marker = "o")
    ax2.hlines(1,1,qaoa.max_depth,color = "blue", ls = "--")

    ax2.tick_params(labelcolor = "blue")
    ax2.set_ylabel(r"Probability $| \langle x^* \vert \gamma \beta \rangle |^2$", color = "blue")

    plt.tight_layout()

    if savefig != None:
        fig.savefig(savefig)
    else:
        plt.show()
