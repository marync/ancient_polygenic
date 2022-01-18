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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab
import copy as cp

# my code
from EqualTheory import *
from plot_simulations_helper import compute_d_ell

# Functions
#-------------------------------------------------------------------------------
def find_threshold (beta, af_function, xvals) :
    """
    Takes an effect size (beta) and function relating effect size to minimum
    allele frequency (af_function) defined over some set of beta values (xvals).
    Returns the minimum allele frequency associated with each value of beta.

    Note that if the site can not be detected as significant, the function
    returns a value of one.
    """

    if beta < np.min (af_function (xvals)) :
        minaf = 1.0
    else :
        minaf = np.min (xnew[np.where (af_function (xvals) - beta < 0)])

    return minaf


# Data
#-------------------------------------------------------------------------------
# Download 'body_HEIGHTz.sumstats.gz' and unzip
# from https://alkesgroup.broadinstitute.org/UKBB

filename = '../data/body_HEIGHTz.sumstats' # summary statistics

# parameters
#-------------------------------------------------------------------------------
ukn             = 458303.                 # sample size
signifThreshold = 1e-6                    # significance threshold
taus            = np.arange (0,1,0.001)   # sampling times in coal. units
aval            = 1e-3                    # pop-scaled mutation rate
outputdir       = '../figures'            # output directory

# read data
#-------------------------------------------------------------------------------
alkes = open (filename, 'r')
alkes = alkes.readlines ()

# create a dictionary which indexes each SNP by chrom and position
alkesDic  = collections.defaultdict (dict)
for i in range(1, len(alkes)) :
#for i in range(1, int(1e6)) :
    line = alkes[i].strip().split()
    chrom = int(line[1])
    pos   = int(line[2])
    beta  = float(line[7])
    af    = float(line[6])
    if af > 0.5 :
        af = 1. - af
    pval  = float(line[9])

    # dictionary values
    alkesDic[(chrom, pos)]['beta'] = beta
    alkesDic[(chrom, pos)]['af']   = af
    alkesDic[(chrom, pos)]['pval'] = pval

# analysis
#-------------------------------------------------------------------------------
# list of significant SNPs and their effect sizes
signif_betas = list ()
signif_afs   = list ()
for key in alkesDic.keys () :
    if alkesDic[key]['pval'] <= signifThreshold :
        signif_betas.append (alkesDic[key]['beta'])
        signif_afs.append (alkesDic[key]['af'])

# discretize effect size distribution: find midpoints
betaDensity = plt.hist (np.abs(signif_betas), bins=1000, density=False)
betaMidPoints = ( (betaDensity[1][:-1] + betaDensity[1][1:]) / 2 )

# Now, we want to find the minimum af needed to detect an effect of the sizes given above
afs = np.linspace (1e-3, 0.5, 1000)

# create dictionary of effect estimates indexed by allele frequency range (rounded)
thresholdDict = collections.OrderedDict ()
roundedafs    = np.round (afs, 4)
for i in range (0, len(afs)-1) :
    thresholdDict[(roundedafs[i], roundedafs[i+1])] = list ()

# iterate through allele frequencies
for i in range (0, len (stringentafs)) :
    diff  = stringent_afs[i] - afs
    left  = np.round (np.where (diff == np.min (diff[np.where (diff > 0)] ))[0], 4)[0]
    right = np.round (np.where (diff == np.max (diff[np.where (diff < 0)] ))[0], 4)[0]
    thresholdDict[(left, right)].append (stringent_betas[i])

# find minimum beta per allele frequency range
minimum_betas = np.zeros (len(afs)-1)
count = 0
for key in thresholdDict.keys () :
    minimum_betas[count] = np.min (np.array(np.abs(thresholdDict[key])))
    count += 1

#for i in range (0, len(afs)-1) :
    #if i % 100 == 0 :
    #    print (i)

#    indices = np.where ( (signif_afs >= afs[i]) & (signif_afs < afs[i+1]))
#    minimum_betas[i] = np.min ( np.array(np.abs(signif_betas))[np.array(indices[0])] )


# To generate a function over all allele frequencies we interpolate between
# allele frequencies
xnew = np.linspace (np.min(afs[:-1]), np.max(afs[:-1]), 2000)

# interpolation
f2 = scipy.interpolate.interp1d (afs[:-1], minimum_betas, kind='linear')

# now find the minimum allele frequency for each beta
#minaf = [np.min(xnew[np.where (f2(xnew) - beta < 0)]) for beta in betaMidPoints]
minaf = [find_threshold (beta=beta, af_function=f2, xvals=xnew) for beta in betaMidPoints]

print ('The minimum effect size that can be detected is: ' + str(np.min(f2(xnew))))


# Find theoretical expectations for statistics
#---------------------------------------------
# compute all of the statistics using the empirical effect size distribution
uniqueaf   = np.unique (minaf) # only need to find for unique afs

# plotting colors
colors = mpl.pylab.cm.jet (np.linspace(np.min(uniqueaf),np.max(uniqueaf),len(uniqueaf)))

# create a dictionary of theory objects
theoryDic  = collections.OrderedDict ()
i = 0
for q in uniqueaf :
    theory_q = EqualTheory (a=aval, d=q*ukn*2, n=ukn, times=taus)
    theory_q.process ()
    theoryDic[q] = cp.copy (theory_q)
    i += 1

# normalize the effect size distribution
normDensity = betaDensity[0] / np.sum (betaDensity[0])

# weight the statistics by the effect size distribution
mses = np.zeros ((len(betaMidPoints), len(taus)))
vas  = np.zeros ((len(betaMidPoints), len(taus)))
for i in range (0, len(betaMidPoints)) :
    mses[i,:] = (betaMidPoints[i]**2) * theoryDic[minaf[i]].mse * normDensity[i]
    vas[i,:]  = (betaMidPoints[i]**2) * theoryDic[minaf[i]].eva * normDensity[i]

# "true" additive genetic variance
uk_trueva = np.sum ( (betaMidPoints**2) * normDensity ) * (aval / (2.*aval + 1.))

# mse and estimated additive genetic variance
ukmse = np.sum ( mses, axis=0 )
ukvas = np.sum ( vas, axis=0 )



## compute the same statistics as if distribution is a point mass at mean
meanbeta = np.mean (np.abs(signif_betas))
print ('mean beta: ' + str(meanbeta))

# find the corresponding minimum af
minafmean = np.min(xnew[np.where (f2(xnew) - meanbeta < 0)])
print ('minimum af for mean: ' + str(minafmean))

# and theory
theory_mean = EqualTheory (a=aval, d=minafmean*ukn*2, n=ukn, times=taus)
theory_mean.process ()

# true and estimated va
uk_truevamean = (meanbeta**2) * (aval / (2.*aval + 1.))
uk_meanva     = (meanbeta**2) * theory_mean.eva

# plots
#-------------------------------------------------------------------------------
# full distribution
fig, axs =  plt.subplots ( 1, 3, figsize=(16,4), sharex=True, sharey=False )

axs[0].hist (np.abs(signif_betas), bins=1000, density=False, color='dodgerblue')
axs[1].scatter (betaMidPoints, betaDensity[0] / np.sum (betaDensity[0]), color='dodgerblue')
axs[2].plot (betaMidPoints, minaf, color='dodgerblue')

axs[0].set_xscale ('log')
axs[0].set_xlabel (r'$|\beta|$')
axs[0].set_ylabel ('count')
axs[0].set_title ('(a) effect size dist.')
axs[1].set_ylabel ('density')
axs[1].set_xlabel (r'$|\beta|$')
axs[1].set_title ('(b) coarse effect size dist.')
axs[2].set_yscale ('log')
axs[2].set_ylabel (r'$z_\beta$')
axs[2].set_xlabel (r'$|\beta|$')
axs[2].set_title ('(c) allele freq.\ threshold')

# save
plt.savefig (os.path.join (outputdir, 'uk_beta_distribution.pdf'), bbox_inches='tight')
plt.close ()

# accuracy (ignoring sample size factor) and relative accuracy with h2 = 0.5
fig, axs =  plt.subplots ( 1, 2, figsize=(12,5), sharex=True )
axs[0].plot (taus, ukvas / (2.*uk_trueva), color='dodgerblue', label='distribution' )
axs[0].plot (taus, uk_meanva / (2.*uk_truevamean), linestyle='--', color='black', label='point mass')
axs[1].plot (taus, uk_meanva / uk_meanva[0], linestyle='--', color='black', label='point mass')
axs[1].plot (taus, ukvas / ukvas[0], color='dodgerblue', label='distribution')

# labels
axs[0].set_xlabel (r'ancient sampling time $\tau$')
axs[0].set_ylabel (r'$\rho^2 (\tau)$')
axs[0].set_title (r'(a) accuracy, $h^2=0.5$')
axs[0].invert_xaxis ()
axs[1].set_ylabel (r'$\rho^2 (\tau) /\ \rho^2 (0)$')
axs[1].set_xlabel (r'ancient sampling time $\tau$')
axs[1].set_title ('(b) rel.\ accuracy, $h^2=0.5$')
axs[0].legend (frameon=False)
axs[1].legend (frameon=False)

# save
plt.savefig (os.path.join (outputdir, 'uk_accuracy.pdf'), bbox_inches='tight')
plt.close ()
