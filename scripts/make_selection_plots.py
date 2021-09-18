import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import copy as cp
import string
import pandas as pd
import collections

# my code
from EqualTheory import *
#from Ancient import *
from plot_simulations_helper import plot_stat, plot_accuracies_a_b

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

# colors
colors = ['#377eb8', '#ff7f00', '#4daf4a',
          '#f781bf', '#a65628', '#984ea3',
          '#999999', '#e41a1c', '#dede00']


#-------------------------------------------------------------------------------
# to produce figure 3
#inputdir  = '../data/heritability'
#h2        = True

# to produce figure 4
inputdir  = '../data/selection'
h2        = False

# shared parameters
a         = 1e-3 # population scaled mut. rate
nsim      = 5000 # number of simulations
preambles = ['d100','d1000'] # sub-directory names referring to thresholds
L         = 5000 # number of loci
N         = 1e3 # population size
n         = 1e4 # gwa study size

# output directory
outputdir = '../figures'

# some global stuff
va         = a / (2.*a + 1.) # additive genetic variance
thresholds = [1000]
dvals      = [int(d) for d in [1,10,100,1e3,5e3,1e4]]

#-------------------------------------------------------------------------------
# input + output

# create directory
if not os.path.exists (outputdir) :
    os.mkdir (outputdir)

# directories of input
if not h2 :
    dirs = [ i for i in os.listdir (inputdir) if os.path.isdir(os.path.join(inputdir,i))]
    numdirs = np.argsort ([float(i) for i in dirs])
    scoeffs = [N*float(dirs[numdirs[i]]) for i in range(len(dirs))]
else :
    dirs = [ i for i in os.listdir (inputdir) if os.path.isdir(os.path.join(inputdir,i))]
    heritabilities = [diri.split ('h2')[1] for diri in dirs]

# load time file
taus = np.loadtxt (os.path.join (inputdir,  'times.csv'), delimiter=',')
theorytaus = np.arange (0,np.max(taus)+1,1)

#-------------------------------------------------------------------------------
#  read simulations

# create dictionary of simulations
simDict = collections.defaultdict (dict)
for i in range(len(dirs)) :
    for j in range (len(preambles)) :
        threshold_j = int(preambles[j].split('d')[1])
        fname = os.path.join(inputdir, dirs[i], preambles[j], 'allsims.csv')
        simDict[dirs[i]][threshold_j] = pd.read_csv (fname, delimiter=',', header=0)

#-------------------------------------------------------------------------------
#  set up plots + output to pdf

if h2 :
    fig, axs = plt.subplots ( 1, 2, figsize=(12,5), sharex=True, sharey=False)
    figname  = 'figure3'
else :
    fig, axs = plt.subplots ( 2, 2, figsize=(10,8), sharex=True, sharey=False)
    figname  = 'figure4'

#  plot
if h2 :
    axs[0] = plot_accuracies_a_b (axs=axs[0], h2=0.5, aval=1e-3, n=1e4, ds=dvals, times=np.arange(0, np.max(taus) / (2.*N), 0.01))
    axs[1] = plot_stat (axs[1], a, n, N, simDict, sigmaprime=[0,va], stat='rho2_trait', times=taus, error=False, nsim=nsim, ylabel=r'$\rho^2 (\tau)$ \& $r^2 (\tau)$', preambles=thresholds, h2=True, sampler2=True, makelegend=False)
    axins = inset_axes(axs[1], width="30%", height="30%", loc=2, borderpad=2.)
    axins = plot_stat (axins, a, n, N, simDict, stat='rho2_trait', sigmaprime=[0,va], relative=True, times=taus, error=False, nsim=nsim, ylabel=None,
            preambles=thresholds, h2=True, sampler2=True)

    axins.locator_params (axis="y", nbins=2)
    axins.invert_xaxis ()
    axins.set_title ('rel.\ accuracy')
    axins.set_xlabel ('')

    axs[1].set_title ('(b) accuracy, varying $h^2$')
    axs[0].invert_xaxis ()

    stat_handles = []
    stat_handles.append(mpl.lines.Line2D([], [], color='black', alpha=0.8, marker='o', linestyle='None', markersize=5))
    stat_handles.append(mpl.lines.Line2D([], [], color='black', alpha=0.8, marker='x', linestyle='None', markersize=5))
    stat_legend = mpl.legend.Legend (axs[0], handles=stat_handles, labels=[r'$\hat \rho^2 (\tau)$', r'$\hat r^2 (\tau)$'], frameon=False, bbox_to_anchor=(-0.15, 0.55), title='statistic')

    h2_handles = []
    h2_handles.append(mpl.lines.Line2D([], [], color='goldenrod', alpha=0.8, marker='s', linestyle='None', markersize=5))
    h2_handles.append(mpl.lines.Line2D([], [], color='dodgerblue', alpha=0.8, marker='s', linestyle='None', markersize=5))
    h2_legend = mpl.legend.Legend (axs[0], handles=h2_handles, labels=[1,0.5], frameon=False, bbox_to_anchor=(-0.15, 0.275), title=r'$h^2$')

    axs[0].add_artist (stat_legend)
    axs[0].add_artist (h2_legend)

else :
    axs[0,0] = plot_stat (axs[0,0], a, n, N, simDict, stat='bias', va=np.sqrt(va), times=taus, makelegend=True, nsim=nsim*L, ylabel=r'$bias_\ell (\tau) / \sqrt{V_{A\ell}}$', preambles=thresholds, error=False, xlabel=False, onset=0.5)
    axs[0,1] = plot_stat (axs[0,1], a, n, N, simDict, stat='mse', va=va, times=taus, nsim=nsim*L, ylabel=r'$mse_\ell (\tau) / V_{A\ell}$', preambles=thresholds, xlabel=False, onset=0.5, error=False)
    axs[1,0] = plot_stat (axs[1,0], a, n, N, simDict, stat='rho2_trait', times=taus, error=False, nsim=nsim, ylabel=r'$\rho^2 (\tau)$', preambles=thresholds, onset=0.5, sampler2=False)
    axs[1,1] = plot_stat (axs[1,1], a, n, N, simDict, stat='rho2_trait', relative=True, times=taus, error=False, nsim=nsim, ylabel=r'$\rho^2 (\tau) / \rho^2 (0)$', preambles=thresholds, onset=0.5, sampler2=False)

    axs[1,0].invert_xaxis ()


# save figure
plt.savefig(os.path.join (outputdir, figname + '.pdf'), bbox_inches='tight')
plt.close ()


