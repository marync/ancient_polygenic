#-------------------------------------------------------------------------------
# Analysis of UK Biobank
#-------------------------------------------------------------------------------

# modules
#-------------------------------------------------------------------------------
import os
import numpy as np
import collections
import scipy
import scipy.special
import scipy.interpolate
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab
import copy as cp

# my code
from EqualTheory import *
from plot_simulations_helper import compute_d_ell, sci_notation
from uk_biobank_helper import find_threshold, folded_normal_mixture

# Data
#-------------------------------------------------------------------------------
# Download 'body_HEIGHTz.sumstats.gz' and unzip
# from https://alkesgroup.broadinstitute.org/UKBB

filename = '../data/body_HEIGHTz.sumstats' # summary statistics

# parameters
#-------------------------------------------------------------------------------
ukn             = 458303.                 # sample size
alpha           = 1e-8                    # stringent significance threshold
taus            = np.arange (0,1,0.001)   # sampling times in coal. units
aval            = 1e-3                    # pop-scaled mutation rate
outputdir       = '../figures'            # output directory
nbins           = 250                     # number of bins by which to discretize afs

# parameters of causal distributions
#-------------------------------------------------------------------------------
expmeans     = [10,100,500] # exponential means
gammascales  = [10**(-3),0.5] # gamma shape parameters
#mixtureprops = [float(1/2), float(1/3), float(1/6)] # mixture proportions for normal
#mixturevars  = [0.001,0.1,1.] # mixture variances for normal
mixtureprops = [0.9,0.1]
mixturevars  = [1.5439*10**(-5),2.021*10**(-4)]

# number of distributions
npdfs  = len(expmeans) + len(gammascales) + 1
colors = ['gold', 'darkorange', 'darkgoldenrod', 'lightskyblue', 'dodgerblue', 'dimgrey'] # should automate this!
labels = ['exp, ' + str(expmeans[0]), 'exp, ' + str(expmeans[1]), 'exp, ' + str(expmeans[2]),
          r'$\Gamma$, ' + str(gammascales[0]), r'$\Gamma$, ' + str(gammascales[1]),
          'mixture']
lines  = [':',':',':','--','--','-']

# beta values at which to discretize
betavals = np.linspace (10**(-4), 0.1, 5000)

# read data
#-------------------------------------------------------------------------------
alkes = open (filename, 'r')
alkes = alkes.readlines ()

# causal distributions
#-------------------------------------------------------------------------------
# compute the discretized distributions
pdfs = np.zeros ((npdfs,len(betavals)))

i = 0 # counter
# exponentials
for mean in expmeans :
    pdfi = mean*np.exp(-mean*betavals)
    pdfs[i,:] = pdfi / np.sum (pdfi)
    i += 1

# gammas
for var in gammascales :
    pdfi = scipy.stats.gamma.pdf (betavals, a=var, loc=0, scale=1.)
    pdfs[i,:] = pdfi / np.sum (pdfi)
    i += 1

# mixture of normals
pdfi = folded_normal_mixture (xs=betavals, mixtureprops=mixtureprops, variances=mixturevars)
pdfs[i,:] = pdfi / np.sum (pdfi)


#print(pdfs[:,-1])

# analysis of summary statistics
#-------------------------------------------------------------------------------
# effect sizes and allele frequencies of SNPs passing the significance
# threshold
stringent_betas  = list ()
stringent_afs    = list ()

# iterate through all of the SNPs
for i in range(1, len(alkes)) :
    line = alkes[i].strip().split()
    beta  = float(line[7])
    af    = float(line[6])
    if af > 0.5 :
        af = 1. - af
    pval  = float(line[9])

    if pval <= alpha:
        stringent_betas.append (beta)
        stringent_afs.append (af)

print ('# total of SNPs passing the significance threshold: ' + str(len(stringent_betas)))

# Now, we want to find the minimum af needed to detect an effect of the sizes given above
#afs = np.linspace (1e-3, 0.5, nbins)
afs = np.logspace (-3, np.log10(0.5), nbins)

# create dictionary of effect estimates indexed by allele frequency range (rounded)
thresholdDict = collections.OrderedDict ()
roundedafs    = np.round (afs, 6)
for i in range (0, len(afs)-1) :
    thresholdDict[(roundedafs[i], roundedafs[i+1])] = list ()

# iterate through allele frequencies
for i in range (0, len (stringent_afs)) :
    diff  = stringent_afs[i] - afs
    left  = np.where (diff == np.min (diff[np.where (diff > 0)]))[0]
    right = np.where (diff == np.max (diff[np.where(diff < 0)]))[0]
    keyi  = (np.round(afs[left],6)[0], np.round(afs[right],6)[0])
    thresholdDict[keyi].append (stringent_betas[i])

# find minimum beta per allele frequency range
minimum_betas = np.zeros (len(afs)-1)
count = 0
for key in thresholdDict.keys () :
    minimum_betas[count] = np.min (np.array(np.abs(thresholdDict[key])))
    count += 1

# interpolation
xnew = np.logspace (np.log10(np.min(afs[:-1])), np.log10(np.max(afs[:-1])), nbins*2)
zbetafunction = scipy.interpolate.interp1d (afs[:-1], minimum_betas, kind='linear')

# now find the minimum allele frequency for each beta
minaf = [find_threshold (beta=beta, af_function=zbetafunction, xvals=xnew) for beta in betavals]

print ('The minimum effect size that can be detected is: ' + str(np.min(zbetafunction(xnew))))
print ('The maximum effect size that was detected is: ' + str(np.max(stringent_betas)))



# Find theoretical expectations for statistics
#---------------------------------------------
# compute all of the statistics using the empirical effect size distribution
uniqueaf = np.sort (np.unique (minaf))

# create a dictionary of theory objects
theoryDic  = collections.OrderedDict ()
for q in uniqueaf :
    if not np.isnan (q) :
        theory_q = EqualTheory (a=aval, d=q*ukn*2, n=ukn, times=taus)
        theory_q.process ()
        theoryDic[q] = cp.copy (theory_q)

# weight the statistics by the effect size distribution
evas  = np.zeros ((npdfs, len(betavals), len(taus)))
for i in range (npdfs) :
    for j in range (len(betavals)) :
        if not np.isnan (minaf[j]) :
            evas[i,j,:]  = (betavals[j]**2) * theoryDic[minaf[j]].eva * pdfs[i,j]
        else :
            evas[i,j,:] = np.zeros (len(taus))


# "true" additive genetic variance
truevas = np.zeros (npdfs)
for i in range (npdfs) :
     truevas[i] = np.sum ( (betavals**2) * pdfs[i,:] ) * (aval / (2.*aval + 1.))

# estimated additive genetic variance
ukevas = np.sum ( evas, axis=1 )


## compute the same statistics as if distribution is a point mass at mean
#meanbeta = np.mean (np.abs(pruned_stringent_betas))
#print ('mean beta: ' + str(meanbeta))

# find the corresponding minimum af
#minafmean = np.min(xnew[np.where (zbetafunction(xnew) - meanbeta < 0)])
#print ('minimum af for mean: ' + str(minafmean))

# and theory
#theory_mean = EqualTheory (a=aval, d=minafmean*ukn*2, n=ukn, times=taus)
#theory_mean.process ()

# true and estimated va
#uk_truevamean = (meanbeta**2) * (aval / (2.*aval + 1.))
#uk_meanva     = (meanbeta**2) * theory_mean.eva

# plots
#-------------------------------------------------------------------------------
# full distribution
fig, axs =  plt.subplots ( 1, 3, figsize=(17.5,4), sharex=False, sharey=False )

#axs[0].hist (np.abs(signif_betas), bins=1000, density=False, color='dodgerblue')
# panel (a) - plot the causal distributions


for i in range(npdfs) :
    axs[0].plot (betavals, pdfs[i,:], linestyle=lines[i], label=labels[i], color=colors[i])


#axs[0].hist (np.abs(stringent_betas), bins=np.logspace( np.log10(np.min(stringent_betas)), np.log10(np.max(stringent_betas)), nbins),
#             density=False, color='dodgerblue', label=sci_notation(alpha))
# panel (b)
axs[1].plot (betavals, minaf, color='black')

# panel (c)
# plot contributions to variance
for i in range(npdfs) :
    axs[2].plot (betavals, evas[i,:,0] / ( (betavals**2) * aval / (2.*aval + 1.)), color=colors[i], label=labels[i])
    axs[2].plot (betavals, pdfs[i,:], linestyle=':', color=colors[i])

axs[2].plot ([np.NAN], [np.NAN], linestyle=':', color='black', label=r'$V_{Ab}$')
axs[2].plot ([np.NAN], [np.NAN], linestyle='-', color='black', label=r'$\hat V_{Ab} (0)$')

# add lines for minimum beta that can be detected
axs[0].axvline(x=np.min(minimum_betas),linewidth=1,linestyle='--',color='black')
axs[1].axvline(x=np.min(minimum_betas),linewidth=1,linestyle='--',color='black')
axs[2].axvline(x=np.min(minimum_betas),linewidth=1,linestyle='--',color='black')

axs[0].set_title ('(a) causal effect distributions')
axs[0].set_xscale ('log')
axs[0].set_xlabel (r'$|\beta|$')
axs[0].set_ylabel (r'$f_\cdot (\cdot)$')
axs[0].legend (markerscale=10, frameon=False)

axs[1].set_title ('(b) allele freq.\ threshold')
axs[1].set_xscale ('log')
axs[1].set_ylabel (r'$g_\alpha (|\beta|)$')
axs[1].set_xlabel (r'$|\beta|$')
axs[1].set_xlim ((0.9*np.min(minimum_betas),np.max(betavals)))

axs[2].set_title ('(c) contribution to variance')
axs[2].set_ylabel (r'norm.\ contrib.\ to $V_A$ or $\hat V_A (0)$')
axs[2].set_xlabel (r'$|\beta|$')
axs[2].set_xscale ('log')
axs[2].set_yscale ('log')
axs[2].legend (frameon=False)

# save
plt.savefig (os.path.join (outputdir, 'uk_beta_distribution.pdf'), bbox_inches='tight')
plt.close ()

# accuracy (ignoring sample size factor) and relative accuracy with h2 = 0.5
fig, axs =  plt.subplots ( 1, 2, figsize=(12,5), sharex=True )

for i in range(npdfs) :
    axs[0].plot (taus, ukevas[i,:] / (2.*truevas[i]), linestyle=lines[i], label=labels[i], color=colors[i] )
    axs[1].plot (taus, ukevas[i,:] / ukevas[i,0], linestyle=lines[i], label=labels[i], color=colors[i] )

# labels
axs[0].set_xlabel (r'ancient sampling time $\tau$')
axs[0].set_ylabel (r'$\rho^2 (\tau)$')
axs[0].set_title (r'(a) accuracy, $h^2=0.5$')
axs[0].invert_xaxis ()
axs[0].legend (frameon=False)

axs[1].set_ylabel (r'$\rho^2 (\tau) /\ \rho^2 (0)$')
axs[1].set_xlabel (r'ancient sampling time $\tau$')
axs[1].set_title ('(b) rel.\ accuracy, $h^2=0.5$')
axs[1].legend (frameon=False)

# save
plt.savefig (os.path.join (outputdir, 'uk_accuracy.pdf'), bbox_inches='tight')
plt.close ()
