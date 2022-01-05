import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import copy as cp
import collections

# my code
from EqualTheory import *

#-------------------------------------------------------------------------------
# Define function for string formatting of scientific notation
# adapted from: https://stackoverflow.com/questions/18311909/how-do-i-annotate-with-power-of-ten-formatting
def sci_notation(num, decimal_digits=1, precision=0, exponent=None):
    """
    Formats number into scientific notation
    """
    if exponent is None:
        exponent = int(np.floor(np.log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    if exponent == 0. :
        out = str(num)
    elif coeff == 1. :
        out = r'$10^{{{0}}}$'.format(exponent)
    else :
        out = r'${0:.{2}f}\cdot10^{{{1:d}}}$'.format(coeff,exponent,precision)

    return out 


def compute_d_ell (n, alpha=1e-8, vp=2, beta=1, vstar=None) :
    """
    Computes the threshold gamma for a given parameter specification.
    """

    if vstar is None :
        vstar  = 2. * scipy.special.erfinv (1. - alpha)**2
        term = 1 - (2. * vstar * (vp / n) / (beta**2))
        if term < 0 :
            gammab = float("NaN")
        else :
            gammab = np.ceil (2.*n* (0.5 - 0.5 * np.sqrt (term)))

    else :
        gammab =  np.round (n * (1. - np.sqrt (1. - ((2.*vstar)/n))))

    return gammab


def compute_fst (a, taus, normalize=False) :
    """
    Computes Fst given a population-scaled mutation rate and sampling / divergence time.
    (Either normalized or not.)
    """

    # compute numerator
    num  = ( (a+1.) / (4.*(2.*a+1.)) )
    num -= ( 1./8. )
    num -= ( 1./ (8.*(2.*a+1.)) ) * np.exp (-a*taus)

    if normalize :
        denom = 0.5 - (((a+1.)/(4.*(2.*a+1.))))
        denom -= (1./8.)
        denom -= (1./8.) * (1. / (2.*a+1.)) * np.exp (-a*taus)
        fst = num / denom

    else :
        fst = num

    return fst


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
    [(10,      (0, (1, 1))),
     (100, (0, (3, 1, 1, 1))),
     (1000,               (0, ())),
     (10000,      (0, (5, 1))),
     (50000, (0, (3, 1, 1, 1, 1, 1))),
     (100000,              (0, (1, 5))),
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
def plot_accuracies_a_b (axs, h2, aval, n, ds, times, fst=False) :
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

    if fst :
        xvals = compute_fst (a=aval, taus=times, normalize=True)
        axs.set_xlabel (r'$F_{ST}$')
    else :
        xvals = cp.copy (times)
        axs.set_xlabel (r'ancient sampling time $\tau$')

    for j in range (len(ds)) :
        linestyle_d = list(linestyles.keys())[j]
        theory = EqualTheory(a=aval, d=ds[j], n=n, times=times)
        theory.process (sigmaprime=sigmaprime)

        #axs.plot (times, theory.rho_tau,
        axs.plot (xvals, theory.rho_tau,
                       color='dodgerblue', alpha=alphas[j], linestyle=linestyles[linestyle_d], lw=1)
        #axins.plot (times, theory.rho_tau / theory.rho_tau[0],
        axins.plot (xvals, theory.rho_tau / theory.rho_tau[0],
                       color='dodgerblue', alpha=alphas[j], linestyle=linestyles[linestyle_d], lw=1)

        custom_lines_ds.append (mpl.lines.Line2D([0], [0], color='black', alpha=alphas[j], lw=1, linestyle=linestyles[linestyle_d]))

    axs.set_title (r'(a) accuracy, varying $d$')
    axins.set_title (r'rel.\ accuracy')
    axs.set_ylabel (r'$\rho^2 (\tau)$')

    axins.invert_xaxis ()
    axins.locator_params (axis="y", nbins=2)
    formatted_ds = [sci_notation(di) for di in ds]
    axs.legend (custom_lines_ds, formatted_ds, title=r'$d$', loc='lower right', bbox_to_anchor=(-0.15,0.5), frameon=False)

    return axs

#-------------------------------------------------------------------------------
def plot_relative_accuracy_data (obs_fst, obs_acc, pred_acc, pops, axs, h2, aval, n, ds, times, fst=False) :
    """
    Plots first part of Figure 3.
    """

    # for heritability
    sigmaprime = (aval / (2.*aval + 1.)) * ((1. - h2) / h2)

    if fst :
        xvals = compute_fst (a=aval, taus=times, normalize=True)
        axs.set_xlabel (r'$F_{ST}$')
    else :
        xvals = cp.copy (times)
        axs.set_xlabel (r'ancient sampling time $\tau$')

    theory = EqualTheory(a=aval, d=ds, n=n, times=times)
    theory.process (sigmaprime=sigmaprime)

    axs.plot (xvals, theory.rho_tau / theory.rho_tau[0],
              color='black', linestyle='--', lw=1)


    # add data
    colors = ['darkorange','dodgerblue','maroon']
    for i in range(len(pops)) :
        axs.scatter (obs_fst[i], obs_acc[i], color=colors[i], marker='o', label=pops[i])
        axs.scatter (obs_fst[i], pred_acc[i], color=colors[i], marker='x')

    axs.set_title (r'rel.\ accuracy')
    axs.set_ylabel (r'$\rho^2 (\tau) /\ \rho^2 (0)$')

    #formatted_ds = [sci_notation(di) for di in ds]
    #axs.legend (custom_lines_ds, formatted_ds, title=r'$d$', loc='lower right', bbox_to_anchor=(-0.15,0.5), frameon=False)
    axs.legend (frameon=False)

    return axs


def plot_stat (ax, a, n, N,  results_dictionary, stat, times, preambles, h2=False, sigmaprime=[0], onset=None, sampler2=False, nsim=None, va=None, relative=False, ylabel=None, xlabel=True, makelegend=False, error=True, jitter=0, fst=False) :
    """
    Plots either selection (figure 4) or (b) of figure 3.
    """

    if h2 :
        colors  = [ 'dodgerblue', 'goldenrod', 'black', 'mediumblue', 'midnightblue', 'green', 'darkgreen']
    else :
        colors  = ['lightslategray', 'lightskyblue', 'dodgerblue', 'mediumblue', 'midnightblue', 'green', 'darkgreen']

    markers = ['o','x','v']

    if stat == 'bias' :
        title = r'(a) $bias_\ell (\tau )$'
    elif stat == 'mse' :
        title = r'(b) $mse_\ell (\tau )$'
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

    if fst :
        xvals = compute_fst (a=a, taus=theorytimes, normalize=True)
        ax.set_xlabel (r'$F_{ST}$')
    else :
        xvals = cp.copy (theorytimes)
        ax.set_xlabel (r'ancient sampling time $\tau$')

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
                #ax.plot (theorytimes, theory / va, color=colors[sigma_counter], linestyle=linestyles[threshold], lw=0.6) 
                ax.plot (xvals, theory / va, color=colors[sigma_counter], linestyle=linestyles[threshold], lw=0.6) 
            else :
                #ax.plot (theorytimes, theory, color=colors[sigma_counter], linestyle=linestyles[linestylelist[2+threshold_counter]], lw=0.6)
                #ax.plot (theorytimes, theory, color=colors[sigma_counter], linestyle=linestyles[threshold], lw=0.6)
                ax.plot (xvals, theory, color=colors[sigma_counter], linestyle=linestyles[threshold], lw=0.6)

            sigma_counter += 1

        selection_counter = 0
        for selectioncoeff in allselection :

            print('selection coeff: ' + str(selectioncoeff) + ', counter: ' + str(selection_counter))
            X = cp.deepcopy( results_dictionary[selectioncoeff][threshold][stat] )
            times_jitter = (times + selection_counter*jitter) / (2*N)
            if fst :
                times_sel  = compute_fst (a=a, taus=times_jitter, normalize=True)
            else :
                times_sel = cp.copy (times_jitter)

            if stat == 'rho2_trait' :
                X *= (99 / 100)

            #times_sel = times

            if relative :
                ax.scatter ( times_sel, X / X[0], color=colors[selection_counter], marker=markers[threshold_counter], alpha=0.7, s=15)
                          #  label='d=' + str(thresholds[k]) + ', s=' + str(selection[i]) )
            elif va is not None :
                ax.scatter ( times_sel, X / va, color=colors[selection_counter], marker=markers[threshold_counter], alpha=0.7, s=15)

            else :
                ax.scatter ( times_sel, X, color=colors[selection_counter], marker=markers[threshold_counter], alpha=0.7, s=15)

            if error :
                V = cp.deepcopy( results_dictionary[selectioncoeff][threshold]['s_' + stat] ) / nsim
                if va is not None :
                    ax.errorbar( times_sel, X / va,
                                 yerr=1.96*np.sqrt(V / (va**2) ), color=colors[selection_counter], linestyle='None', lw=0.5)#lw=0.5)
                else :
                    ax.errorbar( times_sel, X,
                                  yerr=1.96*np.sqrt(V), color=colors[selection_counter], linestyle='None', lw=0.5) #lw=0.5)

            if stat == 'rho2_trait' and sampler2 :

                R2 = cp.deepcopy( results_dictionary[selectioncoeff][threshold]['r2'] )
                VR2 = cp.deepcopy( results_dictionary[selectioncoeff][threshold]['s_r2'] ) / nsim
                if relative :
                    ax.scatter ( times_sel, R2 / R2[0], color=colors[selection_counter], marker='x', alpha=0.8, s=15)
                    ax.errorbar( times_sel, R2 / R2[0],
                                  yerr=1.96*np.sqrt(VR2), color=colors[selection_counter], linestyle='None', lw=0.5)
                else :
                    ax.scatter ( times_sel, R2, color=colors[selection_counter], marker='x', alpha=0.8, s=15)
                    ax.errorbar( times_sel, R2,
                                  yerr=1.96*np.sqrt(VR2), color=colors[selection_counter], linestyle='None', lw=0.5)

            selection_counter += 1
        threshold_counter += 1

    # set axis title
    ax.set_title (title)

    if ylabel is not None :
        ax.set_ylabel (ylabel)

    if xlabel :
        if fst :
            ax.set_xlabel (r'$F_{ST}$')
        else :
            ax.set_xlabel (r'ancient sampling time $\tau$')

    if makelegend :
        handles_d = []
        theories  = []
        for i in range(len(thresholds)) :
            handles_d.append (mpl.lines.Line2D([], [], color='black', alpha=alphas[i], marker=markers[i], linestyle='None',
                                         markersize=5, label=sci_notation(thresholds[i])))
            theories.append (mpl.lines.Line2D([0], [0], linestyle=linestyles[linestylelist[2+i]],
                             label=r'$d=$ ' + sci_notation(thresholds[i]), color='black', linewidth=1))

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

