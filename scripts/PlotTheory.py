import os
import copy as cp
import numpy as np
import scipy
import scipy.special
import matplotlib.pyplot as plt
import matplotlib as mpl
import collections
import pandas as pd


# my code
from EqualTheory import EqualTheory

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
# auxiliary function
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

    if coeff == 1. :
        out = r'$10^{{{0}}}$'.format(exponent)
    else :
        out = r'${0:.{2}f}\cdot10^{{{1:d}}}$'.format(coeff,exponent,precision)

    return out 



#-------------------------------------------------------------------------------
# main body
class PlotTheory :

    # line styles
    linestyles = collections.OrderedDict(
        [('densely dotted',      (0, (1, 1))),
         ('densely dashdotted',  (0, (3, 1, 1, 1))),
         ('solid',               (0, ())),
         ('densely dashed',      (0, (5, 1))),
         ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
         ('dotted',              (0, (1, 5))),
         ('dashdotted',          (0, (3, 5, 1, 5))),
         ('dashed',              (0, (5, 5))),
         ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
         ('loosely dashed',      (0, (5, 10))),
         ('loosely dashdotted',  (0, (3, 10, 1, 10))),
         ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
         ('loosely dotted',      (0, (1, 10)))])

    # markers
    markers = ['o','x','v','*']

    def __init__(self, avals, ns, times, d1=None, d2=None, outputdir='.') :

        # makes warnings exceptions
        np.seterr(all='raise')

        # set values
        self.avals = np.array([float(a) for a in avals])
        self.ns = np.array([int(n) for n in ns])
        self.taus = times
        self.outputdir = str(outputdir)
        self.create_outputdir ()

    def create_outputdir (self) :

        # output directory
        if not os.path.exists(self.outputdir) :
            os.mkdir(self.outputdir)

    def process (self, range_ds, vstar=None) :

        # create dictionary for low values of d
        #dellsLow = np.array([compute_d_ell (n=ni, vp=1, beta=0.575) for ni in self.ns])
        #self.plot_vas_and_mses (n=1e4, ds=dellsLow, extrads=np.array([1,10,100,1000,5000,10000]), mutationrates=self.avals[1], approx=True)
        #self.plot_vas_and_mses (n=1e4, ds=dellsLow, extrads=np.array([1,10,100,1000,5000,10000]), mutationrates=self.avals[1], approx=False)
        dellsLow = np.array([compute_d_ell (n=ni, vp=1, beta=0.1) for ni in self.ns])
        print ('ns: ' + str(self.ns))
        print ('ds: ' + str(dellsLow))
        self.theoryDictLow = self.create_theory_dictionary (ds=dellsLow)
        self.plot_vas_and_mses (n=1e5, ds=dellsLow, extrads=range_ds, mutationrates=self.avals[1], approx=True)
        self.plot_vas_and_mses (n=1e5, ds=dellsLow, extrads=range_ds, mutationrates=self.avals[1], approx=False)

        # compute stats for a higher threshold
        dellsHigh = np.array([compute_d_ell (n=ni, vp=1, beta=.5) for ni in self.ns])
        print ('high ds: ' + str(dellsHigh))
        self.theoryDictHigh = self.create_theory_dictionary (ds=dellsHigh)

        # denser grid
        denseas   = np.logspace (-4, 0, 50)
        densens   = np.logspace (4,6,20)
        denseellsLow  = [compute_d_ell (n=int(ni), vp=1, beta=0.5) for ni in densens]
        denseellsHigh = [compute_d_ell (n=int(ni), vp=1, beta=.1) for ni in densens]

        # create dense dictionaries
        theoryDictDenseLow  = self.create_theory_dictionary (ds=denseellsLow, avals=denseas, nvals=densens)
        theoryDictDenseHigh = self.create_theory_dictionary (ds=denseellsHigh, avals=denseas, nvals=densens)

        # plot normalized and unnormalized heatmaps
        self.plot_mse_and_va_heatmaps (avals=denseas, nvals=densens, theorydictlow=theoryDictDenseLow, theorydicthigh=theoryDictDenseHigh, normalize=False)
        self.plot_mse_and_va_heatmaps (avals=denseas, nvals=densens, theorydictlow=theoryDictDenseLow, theorydicthigh=theoryDictDenseHigh, normalize=True)

        # plot bias figure
        self.plot_asymmetric_bias (avals=[1e-4,1e-3,1e-2,1], d1=1, threshold_as_n=True)
        #self.plot_p3d (avals=[1e-3, 1], ns=densens, ds=np.arange(1,1e4,1), cmap=['Blues', 'Greys'], log=[False, True])


    def plot_mse_and_va_heatmaps (self, avals, nvals, theorydictlow, theorydicthigh, normalize=False) :

        colors = ['dodgerblue', 'darkorange', 'black']

        fig, axs = plt.subplots (2, 3, figsize=(13, 7), sharey=True, sharex=True)
        plt.ticklabel_format (axis='both', style='sci')

        # form aP array
        aPlow, mselow, valow    = self.form_initial_arrays (avals, nvals, theorydictlow, normalize=normalize)
        aPhigh, msehigh, vahigh = self.form_initial_arrays (avals, nvals, theorydicthigh, normalize=normalize)

        vmax = np.max ([np.max(aPlow), np.max(aPhigh)])
        vmin = np.min ([np.min(aPlow), np.min(aPhigh)])

        if normalize :
            logplots = False
        else :
            logplots = True

        pcm1 = self.plot_heatmap (avals=avals, nvals=nvals, X=aPlow, ax=axs[0,0], vmax=vmax, vmin=vmin, log=logplots)
        pcm2 = self.plot_heatmap (avals=avals, nvals=nvals, X=aPhigh, ax=axs[1,0], vmax=vmax, vmin=vmin, log=logplots)
        fig.colorbar(pcm1, ax=axs[0,0])
        fig.colorbar(pcm2, ax=axs[1,0])

        vmax = np.max ([np.max(mselow), np.max(msehigh)])
        vmin = np.min ([np.min(mselow), np.min(msehigh)])

        pcm3 = self.plot_heatmap (avals=avals, nvals=nvals, X=mselow, ax=axs[0,1], vmax=vmax, vmin=vmin, log=True)
        pcm4 = self.plot_heatmap (avals=avals, nvals=nvals, X=msehigh, ax=axs[1,1], vmax=vmax, vmin=vmin, log=True)
        fig.colorbar(pcm3, ax=axs[0,1])
        fig.colorbar(pcm4, ax=axs[1,1])

        pcm5 = self.plot_heatmap (avals=avals, nvals=nvals, X=valow, ax=axs[0,2], log=logplots)
        pcm6 = self.plot_heatmap (avals=avals, nvals=nvals, X=vahigh, ax=axs[1,2], log=logplots)
        fig.colorbar(pcm5, ax=axs[0,2])
        fig.colorbar(pcm6, ax=axs[1,2])

        for i in range (3) :
            axs[1,i].set_xlabel (r'mutation rate, $a$')
            axs[1,i].get_xticklabels()[2].set_color('darkorange')
            axs[1,i].get_xticklabels()[3].set_color('dodgerblue')
            axs[1,i].get_xticklabels()[4].set_color('gold')

        axs[0,0].set_ylabel (r'$n$')
        axs[1,0].set_ylabel (r'$n$')
        axs[0,0].set_xscale ('log')
        axs[0,0].set_yscale ('log')

        if normalize :
            normstring = 'normalized'
        else :
            normstring = ''

        axs[0,0].set_title (r'(a) ' + normstring + r' $2aP^{(d)}$, low $d$')
        axs[1,0].set_title (r'(d) ' + normstring + r' $2aP^{(d)}$, high $d$')
        axs[0,1].set_title (r'(b) ' + normstring + r' $mse_\ell (0)$, low $d$')
        axs[1,1].set_title (r'(e) ' + normstring + r' $mse_\ell (0)$, high $d$')
        axs[0,2].set_title (r'(c) ' + normstring + r' $\hat V_{A\ell} (0)$, low $d$')
        axs[1,2].set_title (r'(f) ' + normstring + r' $\hat V_{A\ell} (0)$, high $d$') 

        fig.tight_layout(pad=.75)
        plt.savefig(os.path.join(self.outputdir, 'initial_mse_va' + normstring + '.pdf'),bbox_inches='tight') 
        plt.close()

 
    def form_initial_arrays (self, avals, nvals, theorydict, normalize=False) :

        aP   = np.zeros ((len(avals), len(nvals)))
        mse0 = np.zeros ((len(avals), len(nvals)))
        va0  = np.zeros ((len(avals), len(nvals)))
        for i in range(len(avals)) :
            a  = avals[i]
            va = a / (2.*a + 1.)
            for j in range(len(nvals)) :
                if normalize :
                    aP[i,j] = avals[i] * theorydict[avals[i]][int(nvals[j])].p3 / va
                    mse0[i,j] = theorydict[avals[i]][int(nvals[j])].mse[0] / va
                    va0[i,j] = theorydict[avals[i]][int(nvals[j])].eva[0] / (va)
                else :
                    aP[i,j] = avals[i] * theorydict[avals[i]][int(nvals[j])].p3
                    mse0[i,j] = theorydict[avals[i]][int(nvals[j])].mse[0]
                    va0[i,j] = theorydict[avals[i]][int(nvals[j])].eva[0]

        return 2.*aP, mse0, va0


    def plot_heatmap (self, avals, nvals, X, ax, vmax=None, vmin=None, log=True) :
        """
        A dense plot of 2aPd
        """

        if log == True :
            pcm = ax.pcolormesh (avals, nvals, np.transpose(X),
                                  cmap='Greys', norm=mpl.colors.LogNorm(vmax=vmax, vmin=vmin), rasterized=True, shading='auto')
        else :
            pcm = ax.pcolormesh (avals, nvals, np.transpose(X),
                                  cmap='Greys', vmax=vmax, vmin=vmin, rasterized=True, shading='auto')

        return pcm


    def plot_vas_and_mses (self, n, ds, extrads, mutationrates=None, approx=False) :
        """
        Plot the six plot figure for mse and eva.
        """

        if mutationrates is None :
            mutationrates = cp.deepcopy(self.avals)

        colors = ['darkorange', 'dodgerblue', 'gold', 'black']
        alphas = np.linspace (0.4,1.,len(self.ns))

        fig, axs = plt.subplots (2, 3, figsize=(13, 7), sharey=False, sharex=True)
        plt.ticklabel_format (axis='both', style='sci')

        axs = self.plot_increase_and_normalized (axs=axs, colors=colors, alphas=alphas, mutationrates=np.array([mutationrates]),approx=approx)
        axs[:,2], lines_d = self.plot_varying_ds (axs=axs[:,2], n=n, extrads=extrads, colors=colors, mutationrates=np.array([mutationrates]), approx=approx)

        # invert x-axis
        axs[0,0].invert_xaxis ()
        for i in range(3) :
            axs[1,i].set_xlabel (r'ancient sampling time $\tau$')

        lines_mutation = []
        for i in range(len(self.avals)) :
            lines_mutation.append (mpl.lines.Line2D([0], [0], color=colors[i], lw=0.75, linestyle='-'))

        # custom n lines
        custom_lines_n = []
        for i in range(len(self.ns)) :
            custom_lines_n.append (mpl.lines.Line2D([0], [0], color='k', marker=self.markers[i], alpha=alphas[i], lw=1.25, linestyle='-'))

        # now add legends
        formatted_ns  = [sci_notation(ni) for ni in self.ns]
        formatted_mus = [sci_notation(ai) for ai in self.avals]
        formatted_ds  = [sci_notation(di) for di in extrads] 

        #legmutation = axs[0,0].legend (lines_mutation, self.avals, title=r'mutation, $a$', bbox_to_anchor=(-0.2,1.0), frameon=False) #prop={'size': 12})
        #legn = axs[0,0].legend (custom_lines_n, self.ns, title=r'$n$', bbox_to_anchor=(-0.2,0.55), frameon=False) #prop={'size': 12}, 
        legmutation = axs[0,0].legend (lines_mutation, formatted_mus, title=r'mutation, $a$', bbox_to_anchor=(-0.2,1.0), frameon=False) #prop={'size': 12})
        legn = axs[0,0].legend (custom_lines_n, formatted_ns, title=r'$n$', bbox_to_anchor=(-0.2,0.55), frameon=False) #prop={'size': 12}, 
        axs[0,0].add_artist(legmutation)
        axs[0,0].add_artist(legn)

        # d legend
        legd = axs[0,0].legend (lines_d, formatted_ds, title=r'$d$', frameon=False, bbox_to_anchor=(-0.2,0.1)) #prop={'size': 12},  
        #legd = axs[0,0].legend (lines_d, extrads, title=r'$d$', frameon=False, bbox_to_anchor=(-0.2,0.1)) #prop={'size': 12},  
        axs[0,0].add_artist(legd)

        # save figure
        fig.tight_layout(pad=0.75)
        plt.savefig(os.path.join(self.outputdir, 'mse_and_va' + '_' + str(approx) + '.pdf'),bbox_inches='tight')
        plt.close()


    def plot_varying_ds (self, axs, n, extrads, colors, mutationrates=None, approx=False) :
        """
        Make panels c and f of figure 2.
        """

        if mutationrates is None :
            mutationrates = cp.deepcopy(self.avals)

        #alphads = np.linspace (0.3,1.,len(extrads))
        lines_d = []
        leg_counter = 0
        for i in range(len(self.avals)) :
            ai     = self.avals[i]
            colori = colors[i]

            if ai in mutationrates :

                for j in range (len(extrads)) :
                    theory = EqualTheory(a=ai, d=extrads[j], n=n, times=self.taus)
                    theory.process ()

                    axs[0].plot (self.taus, theory.mse / (ai / (2.*ai + 1.)), color=colori, label=extrads[j],
                                 linestyle=self.linestyles[list(self.linestyles.keys())[j]], lw=1)
                    axs[1].plot (self.taus, theory.eva / (ai / (2.*ai + 1.)), color=colori, label=extrads[j],
                                 linestyle=self.linestyles[list(self.linestyles.keys())[j]], lw=1)

                    if approx :
                        axs[0].plot (self.taus, theory.mseapprox / (ai / (2.*ai + 1.)), color='black', alpha=0.6, label=extrads[j],
                                     linestyle=self.linestyles[list(self.linestyles.keys())[j]], lw=1)
                        axs[1].plot (self.taus, theory.evaapprox / (ai / (2.*ai + 1.)), color='black', alpha=0.6, label=extrads[j],
                                     linestyle=self.linestyles[list(self.linestyles.keys())[j]], lw=1)

                    if leg_counter < len(extrads) :

                        lines_d.append ( mpl.lines.Line2D([0], [0], color='k', lw=1.,
                                         linestyle=self.linestyles[list(self.linestyles.keys())[j]]) )
                        leg_counter += 1

        # axis labels
        axs[0].set_ylabel (r'$mse_{\ell} (\tau) \//  V_{A\ell} $')
        axs[1].set_ylabel (r'$\hat V_{A\ell} (\tau) \//  V_{A\ell} $')
        axs[1].set_ylim (bottom = -0.01)

        return axs, lines_d


    def plot_increase_and_normalized (self, axs, colors, alphas, mutationrates=None, approx=False) :
        """
        Plot panels a and e of figure 2.
        """

        if mutationrates is None :
            mutationrates = cp.deepcopy(self.avals)

        for i in range (len(self.avals)) :
            colori = colors[i]
            ai     = self.avals[i]
            trueva = ai / (2.*ai + 1.)

            for j in range(len(self.ns)) :
                alphaj = alphas[j]
                nj     = np.int(self.ns[j])
                theoryij = cp.deepcopy(self.theoryDictLow[ai][nj])
                slicej = slice (j*50, len(self.taus), 150)

                # mse increase
                axs[0,0].plot (self.taus, np.abs(theoryij.mse-theoryij.mse[0]), markevery=slicej, marker=self.markers[j], c=colori, alpha=alphaj, label=nj)
                axs[0,0].set_ylabel (r'$\Delta mse_{\ell} (\tau)$') 
                # eva increase
                axs[1,0].plot (self.taus, np.abs(theoryij.eva-theoryij.eva[0]), markevery=slicej, marker=self.markers[j], c=colori, alpha=alphaj, label=nj)
                axs[1,0].set_ylabel (r'$|\Delta \hat V_{A\ell} (\tau)|$') 

                if approx :
                    axs[0,0].plot (self.taus, np.abs(theoryij.mseapprox-theoryij.mseapprox[0]), c=colori, alpha=alphaj, markevery=slicej, marker=self.markers[j], linestyle=':')
                    axs[1,0].plot (self.taus, np.abs(theoryij.evaapprox-theoryij.evaapprox[0]), c=colori, alpha=alphaj, markevery=slicej, marker=self.markers[j], linestyle=':')


                if ai in mutationrates :
                    # plot normalized mse
                    axs[0,1].plot (self.taus, theoryij.mse / trueva, c=colori, markevery=slicej, marker=self.markers[j], alpha=alphaj, label=nj)
                    axs[0,1].set_ylabel (r'$mse_{\ell} (\tau) \//  V_{A\ell} $') 
                    # plot normalized eva
                    axs[1,1].plot(self.taus, theoryij.eva / trueva, c=colori, markevery=slicej, marker=self.markers[j], alpha=alphaj, label=nj)
                    axs[1,1].set_ylabel (r'$\hat V_{A\ell} (\tau) \//  V_{A\ell} $') 

                    if approx :
                        axs[0,1].plot (self.taus, theoryij.mseapprox / trueva, c='black', alpha=alphaj, markevery=slicej, marker=self.markers[j], lw=1)
                        axs[1,1].plot(self.taus, theoryij.evaapprox / trueva, c='black', alpha=alphaj, markevery=slicej, marker=self.markers[j], lw=1)

        # plot titles
        axs[0,0].set_title (r'(a) change in $mse_\ell (\tau)$')
        axs[0,1].set_title (r'(b) normalized $mse_\ell (\tau)$')
        axs[0,2].set_title (r'(c) varying $d$')
        axs[1,0].set_title (r'(d) change in $\hat V_{A\ell}$') 
        axs[1,1].set_title (r'(e) normalized $\hat V_{A\ell} (\tau)$') 
        axs[1,2].set_title (r'(f) varying $d$')

        # yscales
        axs[0,0].set_yscale ('log')
        axs[1,0].set_yscale ('log')

        # y limits
        axs[1,1].set_ylim ((0,1.05))
        axs[1,2].set_ylim ((0,1.05))

        return axs


    def create_theory_dictionary (self, ds, avals=None, nvals=None) :
        """
        Create a dictionary of theory for the small set of mutation rates and ns.
        """

        if avals is None :
            avals = cp.deepcopy (self.avals)
        if nvals is None :
            nvals = cp.deepcopy (self.ns)

        theoryDict = collections.defaultdict (dict)
        for i in range(len(avals)) :
            for j in range(len(nvals)) :
                theory = EqualTheory(a=avals[i], d=ds[j], n=nvals[j], times=self.taus)
                theory.process ()
                theoryDict[avals[i]][int(nvals[j])] = cp.deepcopy(theory)

        return theoryDict


    def compute_sigmaprime (self, h2, a) :
        """
        Computes the scaled environmental variance term, sigma prime.
        """

        num   = 1. - h2
        denom = ((2.*a + 1.) / a) * h2

        return (num / denom)


    def plot_p3d (self, avals, ns, ds, cmap='Greys', out='a', log=False) :
        """
        """

        p3s = np.zeros ((len(avals), len(ds), len(ns)))
        for i in range(len(avals)) :
            for j in range(len(ds)) :
                for k in range(len(ns)) :
                    p3s[i,j,k] = scipy.stats.betabinom.cdf (ds[j], 2.*ns[k], avals[i], avals[i])

        fig, axs = plt.subplots (ncols=1, nrows=2, sharex=True)
        plt.ticklabel_format(axis='both', style='sci')

        for i in range(len(avals)) :
            if log[i] :
                pcm = axs[i].pcolormesh (ns, ds, 2.*p3s[i,:,:], cmap=cmap[i], shading='auto',rasterized=True,
                                   norm=mpl.colors.LogNorm ())
            else :
                pcm = axs[i].pcolormesh (ns, ds, 2.*p3s[i,:,:], cmap=cmap[i], shading='auto', rasterized=True)

            fig.colorbar(pcm, ax=axs[i])
            axs[i].set_xscale ('log')
            axs[i].set_ylabel(r'detection thresholds, $d_\ell$', rotation=90)
            axs[i].set_title (r'$a=$'+ ' ' + str(avals[i]))

        axs[1].set_xlabel(r'GWA study size, $n$')
        plt.savefig(os.path.join(self.outputdir, 'p3_' + str(out) + '.pdf'),
                    bbox_inches='tight')
        plt.close()


    def plot_asymmetric_bias (self, avals, d1, d2=None, threshold_as_n=False) :
        """
        Plots asymmetric bias.
        """

        colors = ['darkorange', 'dodgerblue', 'goldenrod', 'grey']
        alphas = np.linspace (0.4,1.,len(self.ns))

        # d2 values
        fig, ax = plt.subplots ()

        for i in range(len(avals)) :
            a = avals[i]
            for j in range(len(self.ns)) :
                n = self.ns[j]
                if threshold_as_n :
                    d2 = n

                # for marker location
                slicej = slice (j*50, len(self.taus), len(self.ns)*50)

                # compute theory
                theoryasym = EqualTheory (a=a, d=d1, d2=d2, n=n, times=self.taus)
                theoryasym.process ()

                # normalize by va
                va = np.sqrt (a / (2.*a + 1.))
                ax.plot(self.taus, theoryasym.bias / va, c=colors[i], alpha=alphas[j], markevery=slicej, marker=self.markers[j], label=np.int(self.ns[j]))

        # axis labeling
        ax.set_ylabel(r'$bias_\ell (\tau)$ normalized by $\sqrt{V_{A\ell}}$', rotation=90)
        ax.set_yscale ('log')
        ax.invert_xaxis ()

        if threshold_as_n :
            ax.set_title (r'$d_{\ell 1}=$ ' + str(d1) + ', ' + r'$d_{\ell 2}=n$')
        else :
            ax.set_title (r'$d_{\ell 1}=$' + str(d1) + ', ' + r'$d_{\ell 2}=$' + str(d2))

        # custom n lines
        custom_lines_n = []
        for i in range(len(self.ns)) :
            custom_lines_n.append (mpl.lines.Line2D([0], [0], color='k', marker=self.markers[i], alpha=alphas[i], lw=1.25, linestyle='-'))

        lines_mutation = []
        for i in range(len(avals)) :
            lines_mutation.append (mpl.lines.Line2D([0], [0], color=colors[i], lw=0.75, linestyle='-'))

        legmutation = ax.legend (lines_mutation, avals, title=r'mutation, $a$', prop={'size': 10}, bbox_to_anchor=(-0.15,0.8), frameon=False)
        legn = ax.legend (custom_lines_n, self.ns, title=r'$n$', prop={'size': 10}, bbox_to_anchor=(-0.15,0.55), frameon=False)
        ax.add_artist(legmutation)
        ax.add_artist(legn)

        ax.set_xlabel(r'ancient sampling time $\tau$')
        plt.savefig(os.path.join(self.outputdir, 'asymm_bias.pdf'), bbox_inches='tight')
        plt.close()

