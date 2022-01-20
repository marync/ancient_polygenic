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
from plot_simulations_helper import compute_d_ell, sci_notation
from uk_biobank_helper import find_threshold, prune_snps

# Data
#-------------------------------------------------------------------------------
# Download 'body_HEIGHTz.sumstats.gz' and unzip
# from https://alkesgroup.broadinstitute.org/UKBB

filename = '../data/body_HEIGHTz.sumstats' # summary statistics

# parameters
#-------------------------------------------------------------------------------
ukn             = 458303.                 # sample size
alpha           = 1e-3                    # lax significance threshold
alphaprime      = 1e-8                    # stringent significance threshold
taus            = np.arange (0,1,0.001)   # sampling times in coal. units
aval            = 1e-3                    # pop-scaled mutation rate
outputdir       = '../figures'            # output directory
mindistance     = 0.5e5                   # minimum distance between SNPs in pruned set
nbins           = 500                     # number of bins by which to discretize

# read data
#-------------------------------------------------------------------------------
alkes = open (filename, 'r')
alkes = alkes.readlines ()

# create a dictionary which indexes each SNP by chrom and position
#alkesDic  = collections.defaultdict (dict)
alkesDic = collections.OrderedDict ()

for i in range(1, len(alkes)) :
    line = alkes[i].strip().split()
    chrom = int(line[1])
    pos   = int(line[2])
    beta  = float(line[7])
    af    = float(line[6])
    if af > 0.5 :
        af = 1. - af
    pval  = float(line[9])

    # check for chrom key in dictionary
    if chrom not in alkesDic.keys () :
        alkesDic[chrom] = dict ()

    # add position to dictionary
    alkesDic[chrom][pos] = dict ()
    alkesDic[chrom][pos]['beta'] = beta
    alkesDic[chrom][pos]['af']   = af
    alkesDic[chrom][pos]['pval'] = pval

    # dictionary values
    #alkesDic[(chrom, pos)]['beta'] = beta
    #alkesDic[(chrom, pos)]['af']   = af
    #alkesDic[(chrom, pos)]['pval'] = pval

# analysis
#-------------------------------------------------------------------------------
# create dictionaries containing SNPs filtered at two significance thresholds
signifDic          = collections.OrderedDict ()
signifStringentDic = collections.OrderedDict ()

# more stringent significance threshold
stringent_betas  = list ()
stringent_afs    = list ()

# alpha snps counter
nalphasnps = 0

for chrom in alkesDic.keys () :
    signifDic[chrom]          = dict ()
    signifStringentDic[chrom] = dict ()
    for pos in alkesDic[chrom].keys () :
        pvali = alkesDic[chrom][pos]['pval']

        if pvali <= alpha:
            # add to lax dictionary
            signifDic[chrom][pos] = cp.deepcopy (alkesDic[chrom][pos])
            nalphasnps += 1

            if pvali <= alphaprime :
                # add to stringent dictionary
                signifStringentDic[chrom][pos] = cp.deepcopy (alkesDic[chrom][pos])
                # for convenience, add to these lists
                stringent_betas.append (alkesDic[chrom][pos]['beta'])
                stringent_afs.append (alkesDic[chrom][pos]['af'])

# now, conduct pruning on lax and stringent sets
pruned_indices, pruned_betas                     = prune_snps (snpdict=signifDic,
                                                               distance=mindistance)
pruned_stringent_indices, pruned_stringent_betas = prune_snps (snpdict=signifStringentDic,
                                                               distance=mindistance)

print ('# total of lax SNPS: ' + str(nalphasnps))
print ('# of pruned lax SNPs: ' + str(len(pruned_betas)))
print ('# total of stringent SNPS: ' + str(len(stringent_betas)))
print ('# of pruned stringent SNPs: ' + str(len(pruned_stringent_betas)))


# list of significant SNPs and their effect sizes
#signif_betas = list ()
#signif_afs   = list ()
#for key in alkesDic.keys () :
#    if alkesDic[key]['pval'] <= signifThreshold :
#        signif_betas.append (alkesDic[key]['beta'])
#        signif_afs.append (alkesDic[key]['af'])

# discretize effect size distribution: find midpoints for lax and stringent
# lax
minpruned = np.min (np.abs (pruned_betas))
maxpruned = np.max (np.abs (pruned_betas))
betaDensity   = plt.hist (np.abs(pruned_betas), density=False,
                          bins=np.logspace( np.log10(minpruned), np.log10(maxpruned), nbins))
betaMidPoints = ( (betaDensity[1][:-1] + betaDensity[1][1:]) / 2 )
# stringent
minprunedstringent = np.min (np.abs (pruned_stringent_betas))
maxprunedstringent = np.max (np.abs (pruned_stringent_betas))
betaStringentDensity   = plt.hist (np.abs(pruned_stringent_betas), density=False,
                                   bins=np.logspace( np.log10(minprunedstringent), np.log10(maxprunedstringent), nbins ))
betaStringentMidPoints = ( (betaStringentDensity[1][:-1] + betaStringentDensity[1][1:]) / 2 )

# Now, we want to find the minimum af needed to detect an effect of the sizes given above
#afs = np.linspace (1e-3, 0.5, nbins)
afs = np.logspace (-3, np.log10(0.5), nbins)

# create dictionary of effect estimates indexed by allele frequency range (rounded)
thresholdDict = collections.OrderedDict ()
roundedafs    = np.round (afs, 4)
for i in range (0, len(afs)-1) :
    thresholdDict[(roundedafs[i], roundedafs[i+1])] = list ()

# iterate through allele frequencies
for i in range (0, len (stringent_afs)) :
    diff  = stringent_afs[i] - afs
    #left  = np.round (np.where (diff == np.min (diff[np.where (diff > 0)] ))[0], 4)[0]
    #right = np.round (np.where (diff == np.max (diff[np.where (diff < 0)] ))[0], 4)[0]
    left  = np.where (diff == np.min (diff[np.where (diff > 0)]))[0]
    right = np.where (diff == np.max (diff[np.where(diff < 0)]))[0]
    keyi  = (np.round(afs[left],4)[0], np.round(afs[right],4)[0])
    thresholdDict[keyi].append (stringent_betas[i])
    #thresholdDict[(left, right)].append (stringent_betas[i])

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
#xnew = np.linspace (np.min(afs[:-1]), np.max(afs[:-1]), 2000)
xnew = np.logspace (np.log10(np.min(afs[:-1])), np.log10(np.max(afs[:-1])), nbins*2)

# interpolation
zbetafunction = scipy.interpolate.interp1d (afs[:-1], minimum_betas, kind='linear')

# now find the minimum allele frequency for each beta
minaf          = [find_threshold (beta=beta, af_function=zbetafunction, xvals=xnew) for beta in betaMidPoints]
minafStringent = [find_threshold (beta=beta, af_function=zbetafunction, xvals=xnew) for beta in betaStringentMidPoints]

print ('The minimum effect size that can be detected is: ' + str(np.min(zbetafunction(xnew))))


# Find theoretical expectations for statistics
#---------------------------------------------
# compute all of the statistics using the empirical effect size distribution
#uniqueaf   = np.unique (minaf) # only need to find for unique afs
uniqueaf = np.sort (np.append( np.unique (minaf), np.unique(minafStringent)))

# plotting colors
#colors = mpl.pylab.cm.jet (np.linspace(np.min(uniqueaf),np.max(uniqueaf),len(uniqueaf)))

# create a dictionary of theory objects
theoryDic  = collections.OrderedDict ()
for q in uniqueaf :
    if q != 1 :
        theory_q = EqualTheory (a=aval, d=q*ukn*2, n=ukn, times=taus)
        theory_q.process ()
        theoryDic[q] = cp.copy (theory_q)

# normalize the effect size distribution
normDensity = betaDensity[0] / np.sum (betaDensity[0])
normStringentDensity = betaStringentDensity[0] / np.sum (betaStringentDensity[0])

# weight the statistics by the effect size distribution
mses = np.zeros ((len(betaMidPoints), len(taus), 2))
vas  = np.zeros ((len(betaMidPoints), len(taus), 2))
for i in range (0, len(betaMidPoints)) :
    if minaf[i] != 1 :
        mses[i,:,0] = (betaMidPoints[i]**2) * theoryDic[minaf[i]].mse * normDensity[i]
        vas[i,:,0]  = (betaMidPoints[i]**2) * theoryDic[minaf[i]].eva * normDensity[i]
    else :
        mses[i,:,0] = np.zeros (len(taus))
        vas[i,:,0]  = np.zeros (len(taus))

for i in range (0, len(betaStringentMidPoints)) :
    if minafStringent[i] != 1 :
        mses[i,:,1] = (betaStringentMidPoints[i]**2) * theoryDic[minafStringent[i]].mse * normStringentDensity[i]
        vas[i,:,1]  = (betaStringentMidPoints[i]**2) * theoryDic[minafStringent[i]].eva * normStringentDensity[i]
    else :
        mses[i,:,0] = np.zeros (len(taus))
        vas[i,:,0]  = np.zeros (len(taus))

# "true" additive genetic variance
uk_trueva    = np.zeros (2)
uk_trueva[0] = np.sum ( (betaMidPoints**2) * normDensity ) * (aval / (2.*aval + 1.))
uk_trueva[1] = np.sum ( (betaStringentMidPoints**2) * normStringentDensity ) * (aval / (2.*aval + 1.))

# mse and estimated additive genetic variance
ukmse = np.sum ( mses, axis=0 )
ukvas = np.sum ( vas, axis=0 )


## compute the same statistics as if distribution is a point mass at mean
meanbeta = np.mean (np.abs(pruned_stringent_betas))
print ('mean beta: ' + str(meanbeta))

# find the corresponding minimum af
minafmean = np.min(xnew[np.where (zbetafunction(xnew) - meanbeta < 0)])
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

#axs[0].hist (np.abs(signif_betas), bins=1000, density=False, color='dodgerblue')
# panel (a)
axs[0].hist (np.abs(pruned_betas), bins=np.logspace( np.log10(minpruned), np.log10(maxpruned), nbins),
             density=False, color='dodgerblue', label=sci_notation(alpha))
axs[0].hist (np.abs(pruned_stringent_betas), bins=np.logspace( np.log10(minpruned), np.log10(maxpruned), nbins),
             density=False, color='orange', label=sci_notation(alphaprime))
# panel (b)
axs[1].scatter (betaMidPoints, betaDensity[0] / np.sum (betaDensity[0]), color='dodgerblue', marker='v',
                label=sci_notation(alpha), s=5)
axs[1].scatter (betaStringentMidPoints, betaStringentDensity[0] / np.sum (betaStringentDensity[0]), color='orange',
                marker='1', label=sci_notation(alphaprime), s=5)
# panel (c)
axs[2].plot (betaMidPoints, minaf, color='dodgerblue', label=sci_notation(alpha))
axs[2].plot (betaStringentMidPoints, minafStringent, color='orange', label=sci_notation(alphaprime))

axs[0].set_xscale ('log')
axs[0].set_xlabel (r'$|\beta|$')
axs[0].set_ylabel ('count')
axs[0].set_title ('(a) effect size dist.')
axs[0].legend (title=r'$\alpha$', frameon=False)
axs[1].set_ylabel ('density')
axs[1].set_xlabel (r'$|\beta|$')
axs[1].set_title ('(b) coarse, pruned effect size dist.')
axs[1].legend (title=r'$\alpha$', frameon=False, markerscale=4)
axs[2].set_yscale ('log')
axs[2].set_ylabel (r'$z_\beta$')
axs[2].set_xlabel (r'$|\beta|$')
axs[2].set_title ('(c) allele freq.\ threshold')
axs[2].legend (title=r'$\alpha$', frameon=False)

# save
plt.savefig (os.path.join (outputdir, 'uk_beta_distribution.pdf'), bbox_inches='tight')
plt.close ()

# accuracy (ignoring sample size factor) and relative accuracy with h2 = 0.5
fig, axs =  plt.subplots ( 1, 2, figsize=(12,5), sharex=True )
axs[0].plot (taus, ukvas[:,0] / (2.*uk_trueva[0]), color='dodgerblue', label=r'$\alpha=$'+' '+ sci_notation(alpha) )
axs[0].plot (taus, ukvas[:,1] / (2.*uk_trueva[1]), color='orange', label=r'$\alpha=$'+' '+sci_notation(alphaprime) )
axs[0].plot (taus, uk_meanva / (2.*uk_truevamean), linestyle='--', color='black', label=r'$\beta=$'+' '+str(np.round(meanbeta,3)))
axs[1].plot (taus, ukvas[:,0] / ukvas[0,0], color='dodgerblue', label=r'$\alpha=$'+' '+sci_notation(alpha))
axs[1].plot (taus, ukvas[:,1] / ukvas[0,1], color='orange', label=r'$\alpha=$'+' '+sci_notation(alphaprime) )
axs[1].plot (taus, uk_meanva / uk_meanva[0], linestyle='--', color='black', label=r'$\beta=$'+' '+str(np.round(meanbeta,3)))

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
