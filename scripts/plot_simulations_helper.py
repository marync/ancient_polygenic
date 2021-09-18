import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import copy as cp
import collections

# my code
from EqualTheory import *
#-------------------------------------------------------------------------------
# plotting parameters

mpl.rcParams['text.usetex'] = True
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

params = {'legend.fontsize': 14,
          'legend.title_fontsize':14,
          'font.size': 14,
          'axes.labelsize': 14,
          'axes.titlesize': 14,
          'xtick.labelsize': 11,
          'ytick.labelsize': 11}
mpl.rcParams.update(params)

#-------------------------------------------------------------------------------
# plot features

linestyles = collections.OrderedDict(
    [(1,      (0, (1, 1))),
     (10, (0, (3, 1, 1, 1))),
     (100,               (0, ())),
     (1000,      (0, (5, 1))),
     (5000, (0, (3, 1, 1, 1, 1, 1))),
     (10000,              (0, (1, 5))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('dashed',              (0, (5, 5))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashed',      (0, (5, 10))),
     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('loosely dotted',      (0, (1, 10)))])

# list of line styles
linestylelist = list(linestyles.keys())

# markers
markers = ['o','x','v']
colors  = ['dodgerblue', 'mediumturquoise', 'gold', 'darkorange']

#-------------------------------------------------------------------------------
def plot_accuracies_a_b (axs, h2, aval, n, ds, times) :
    """
    Plots first part of Figure 3.
    """

    # inset formatting
    axins = inset_axes(axs, width="30%", height="30%", loc=2, borderpad=2.)

    # for heritability
    sigmaprime = (aval / (2.*aval + 1.)) * ((1. - h2) / h2)
    alphas = np.flip(np.linspace (0.9,1.0,len(ds)))

    # for custom legends
    custom_lines = []
    custom_lines_ds = []

    for j in range (len(ds)) :
        linestyle_d = list(linestyles.keys())[j]
        theory = EqualTheory(a=aval, d=ds[j], n=n, times=times)
        theory.process (sigmaprime=sigmaprime)

        axs.plot (times, theory.rho_tau,
                       color='dodgerblue', alpha=alphas[j], linestyle=linestyles[linestyle_d], lw=1)
        axins.plot (times, theory.rho_tau / theory.rho_tau[0],
                       color='dodgerblue', alpha=alphas[j], linestyle=linestyles[linestyle_d], lw=1)

        custom_lines_ds.append (mpl.lines.Line2D([0], [0], color='black', alpha=alphas[j], lw=1, linestyle=linestyles[linestyle_d]))

    axs.set_title (r'(a) accuracy, varying $d$')
    axins.set_title (r'rel.\ accuracy')

    axs.set_ylabel (r'$\rho^2 (\tau)$')
    axs.set_xlabel (r'ancient sampling time $\tau$')

    axins.invert_xaxis ()
    axins.locator_params (axis="y", nbins=2)
    axs.legend (custom_lines_ds, ds, title=r'$d$', loc='lower right', bbox_to_anchor=(-0.15,0.5), frameon=False)

    return axs



def plot_stat (ax, a, n, N,  results_dictionary, stat, times, preambles, h2=False, sigmaprime=[0], onset=None, sampler2=False, nsim=None, va=None, relative=False, ylabel=None, xlabel=True, makelegend=False, error=True, jitter=0) :
    """
    Plots either selection (figure 4) or (b) of figure 3.
    """

    if h2 :
        colors  = [ 'dodgerblue', 'goldenrod', 'black', 'mediumblue', 'midnightblue', 'green', 'darkgreen']
    else :
        colors  = ['lightslategray', 'lightskyblue', 'dodgerblue', 'mediumblue', 'midnightblue', 'green', 'darkgreen']

    markers = ['o','x','v']

    if stat == 'bias' :
        title = r'(a) $bias (\tau )$'
    elif stat == 'mse' :
        title = r'(b) $mse (\tau )$'
    elif (stat == 'rho2_trait' or stat == 'rho_tau') and not relative :
        title = r'(c) accuracy'
    elif stat == 'rho2_trait' and relative :
        title = r'(d) relative accuracy'
    if va is not None :
        title += r' normalized'

    # get the theory
    testkey      = list(results_dictionary.keys())[0]
    allselection = np.sort(list(results_dictionary.keys()))
    thresholds   = np.sort(list(results_dictionary[testkey].keys()))
    alphas       = np.flip(np.linspace (0.9,1.0,len(thresholds)))

    # theory times
    theorytimes  = np.arange (0, np.max(times)+1, 1) / (2.*N)
    theorytimes += (len(allselection)*0.5*jitter / (2.*N))

    threshold_counter = 0
    for threshold in results_dictionary[testkey].keys() :

        theoryObj = EqualTheory (a=a, d=threshold, n=n, times=theorytimes)
        sigma_counter = 0
        for sigma in np.flip(np.sort(sigmaprime)) :
            theoryObj.process (sigmaprime=sigma)

            if stat == 'bias' :
                theory = theoryObj.bias
            elif stat == 'mse' :
                theory = theoryObj.mse
            elif (stat == 'rho2_trait' or stat == 'rho_tau') and not relative :
                theory = theoryObj.rho_tau
            elif (stat == 'rho2_trait' or stat == 'rho_tau') and relative :
                theory = theoryObj.rho_tau / theoryObj.rho_tau[0]

            # theory times in middle
            if va is not None :
                #ax.plot (theorytimes, theory / va, color=colors[sigma_counter], linestyle=linestyles[linestylelist[2+threshold_counter]], lw=0.6) 
                ax.plot (theorytimes, theory / va, color=colors[sigma_counter], linestyle=linestyles[threshold], lw=0.6) 
            else :
                #ax.plot (theorytimes, theory, color=colors[sigma_counter], linestyle=linestyles[linestylelist[2+threshold_counter]], lw=0.6)
                ax.plot (theorytimes, theory, color=colors[sigma_counter], linestyle=linestyles[threshold], lw=0.6)

            sigma_counter += 1

        selection_counter = 0
        for selectioncoeff in allselection :

            print('selection coeff: ' + str(selectioncoeff) + ', counter: ' + str(selection_counter))
            X = cp.deepcopy( results_dictionary[selectioncoeff][threshold][stat] )
            times_sel = (times + selection_counter*jitter)

            if stat == 'rho2_trait' :
                X *= (99 / 100)

            #times_sel = times

            if relative :
                ax.scatter ( times_sel / (2*N), X / X[0], color=colors[selection_counter], marker=markers[threshold_counter], alpha=0.7, s=15)
                          #  label='d=' + str(thresholds[k]) + ', s=' + str(selection[i]) )
            elif va is not None :
                ax.scatter ( times_sel / (2*N), X / va, color=colors[selection_counter], marker=markers[threshold_counter], alpha=0.7, s=15)

            else :
                ax.scatter ( times_sel / (2*N), X, color=colors[selection_counter], marker=markers[threshold_counter], alpha=0.7, s=15)

            if error :
                V = cp.deepcopy( results_dictionary[selectioncoeff][threshold]['s_' + stat] ) / nsim
                if va is not None :
                    ax.errorbar( (times_sel) / (2*N), X / va,
                                 yerr=1.96*np.sqrt(V / (va**2) ), color=colors[selection_counter], linestyle='None', lw=0.5)#lw=0.5)
                else :
                    ax.errorbar( (times_sel) / (2*N), X,
                                  yerr=1.96*np.sqrt(V), color=colors[selection_counter], linestyle='None', lw=0.5) #lw=0.5)

            if stat == 'rho2_trait' and sampler2 :

                R2 = cp.deepcopy( results_dictionary[selectioncoeff][threshold]['r2'] )
                VR2 = cp.deepcopy( results_dictionary[selectioncoeff][threshold]['s_r2'] ) / nsim
                if relative :
                    ax.scatter ( (times_sel) / (2*N), R2 / R2[0], color=colors[selection_counter], marker='x', alpha=0.8, s=15)
                    ax.errorbar( (times_sel) / (2*N), R2 / R2[0],
                                  yerr=1.96*np.sqrt(VR2), color=colors[selection_counter], linestyle='None', lw=0.5)
                else :
                    ax.scatter ( (times_sel) / (2*N), R2, color=colors[selection_counter], marker='x', alpha=0.8, s=15)
                    ax.errorbar( (times_sel) / (2*N), R2,
                                  yerr=1.96*np.sqrt(VR2), color=colors[selection_counter], linestyle='None', lw=0.5)

            selection_counter += 1
        threshold_counter += 1

    # set axis title
    ax.set_title (title)

    if ylabel is not None :
        ax.set_ylabel (ylabel)

    if xlabel :
        ax.set_xlabel (r'ancient sampling time $\tau$')

    if makelegend :
        handles_d = []
        theories  = []
        for i in range(len(thresholds)) :
            handles_d.append (mpl.lines.Line2D([], [], color='black', alpha=alphas[i], marker=markers[i], linestyle='None',
                                         markersize=5, label=thresholds[i]))
            theories.append (mpl.lines.Line2D([0], [0], linestyle=linestyles[linestylelist[2+i]],
                             label=r'$d=$ ' + str(thresholds[i]), color='black', linewidth=1))

        legend_d = ax.legend(handles=handles_d, title=r'$d$', frameon=False, bbox_to_anchor=(-0.25,0.8))

        handles = []
        for i in range(len(allselection)) :
            handles.append(mpl.lines.Line2D([], [], color=colors[i], alpha=alphas[0], marker=markers[0], linestyle='None', markersize=5))

        if h2 :
            title_selection = r'$h^2$'
            ns_legend = mpl.legend.Legend(ax, handles=handles, labels=[allsel.split ('h2')[1] for allsel in allselection], title=title_selection,
                                          bbox_to_anchor=(-0.25,0.55), frameon=False)
        else :
            title_selection = r'$4Ns$'
            ns_legend = mpl.legend.Legend(ax, handles=handles, labels=allselection, title=title_selection,
                                          bbox_to_anchor=(-0.25,0.45), frameon=False)
            legend_theory = ax.legend (handles=theories,title='neutral \n theory', frameon=False, bbox_to_anchor=(-0.2,-0.05))
            ax.add_artist (legend_theory)

        ax.add_artist (ns_legend)
        ax.add_artist (legend_d)

    if onset is not None :
        ax.axvline(x = onset, color = 'gray', lw = 2, alpha=0.5)

    return ax

