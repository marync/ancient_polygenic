import numpy as np
import copy as cp
import os

import matplotlib.pyplot as plt
import scipy
from scipy import stats

from Gwas import *
from Ancient import *

from collections import Counter


class SamplingLarge (Gwas, Ancient) :


    def __init__ (self, ploidy, N, gwasSize, L, beta, epsilon, tau, sigma=0., offset=0.) :

        self.N            = float(ploidy) * float(N)
        self.n            = int(gwasSize)
        self.epsilon      = int(epsilon)
        self.beta         = float(beta)
        self.L            = int(L)
        self.ploidy       = int(ploidy)
        self.offset       = int(offset)
        self.sigma        = float(sigma)
        self.tau          = int(tau)


    def simulate (self, time, nsim, ancient, contemp, seed, na=100, lower=None, upper=None, outputdir='.') :

        np.random.seed (int(seed))

        L = ancient.shape[0] # number of loci
        K = nsim*L # number of single locus simulations

        # where to store locus information
        ps_ell     = np.zeros ((L, nsim))
        true_ell   = np.zeros ((L, nsim))
        mups_ell   = np.zeros ((L, nsim))
        mutrue_ell = np.zeros ((L, nsim))
        eva_ell    = np.zeros ((L, nsim))
        va_ell     = np.zeros ((L, nsim))
        contrue    = np.zeros ((L, nsim))
        conps      = np.zeros ((L, nsim))
        xbars      = np.zeros ((L, nsim))
        bhats_mat  = np.zeros ((L, nsim))

        # ancient correlation
        ancient_correlations = np.zeros (nsim)
        ancient_cov          = np.zeros (nsim)
        ancient_varyhat      = np.zeros (nsim)
        ancient_vary         = np.zeros (nsim)

        for i in range (nsim) :

            anci = cp.deepcopy (ancient[:,i])
            coni = cp.deepcopy (contemp[:,i])

            # gwas and ancient sampling
            betahats, chat, xbars_i = self.conduct_gwas (frequencies = coni, lower=lower, upper=upper)
            genotype, phenotype     = self.sample_ancient (frequencies = anci)

            # for rho
            ps_ell[:,i]     = genotype*betahats
            true_ell[:,i]   = cp.deepcopy (genotype)
            mups_ell[:,i]   = (2.*anci - 1.) * betahats
            mutrue_ell[:,i] = (2.*anci - 1.)
            eva_ell[:,i]    = anci * (1. - anci) * betahats
            va_ell[:,i]     = anci * (1. - anci)
            contrue[:,i]    = (2.*coni - 1.)
            conps[:,i]      = (2.*coni - 1.) * betahats

            # for bias and mse
            xbars[:,i] = cp.deepcopy (xbars_i)
            bhats_mat[:,i] = cp.deepcopy (betahats)

            # ancient sample correlation
            ancient_yhats, ancient_ys = self.sample_na_ancients (frequencies=anci, na=na, betas=betahats, intercept=chat)
            ancient_correlations[i]   = self.estimate_ancient_correlation (yhat=ancient_yhats, y=ancient_ys)
            ancient_cov[i]     = np.cov (ancient_yhats, ancient_ys)[0,1]
            ancient_varyhat[i] = np.var (ancient_yhats)
            ancient_vary[i]    = np.var (ancient_ys)

        # for bias
        biasmat = (xbars - true_ell )*(1. - bhats_mat)
        bias    = np.mean ( biasmat )
        s_bias  = ( K / (K-1.) ) * (np.mean (biasmat**2) - bias**2)

        # mse
        msemat = ((xbars - true_ell )**2) * ((1. - bhats_mat)**2)
        mse    = np.mean ( msemat )
        s_mse  = ( K / (K-1.) ) * (np.mean (msemat**2) - mse**2)

        # rho squared
        crossmoment = np.mean ((ps_ell - mups_ell)*(true_ell - mutrue_ell))
        meanps      = np.mean (ps_ell - mups_ell)
        meantrue    = np.mean (true_ell - mutrue_ell)
        innersps    = (ps_ell - mups_ell)**2
        innersphen  = (true_ell - mutrue_ell)**2

        # now covariance
        covariance   = crossmoment - (meanps*meantrue)
        varianceps   = np.mean (innersps) - (meanps)**2
        variancetrue = np.mean (innersphen) - (meantrue)**2
        rho_tau = covariance**2 / (varianceps*variancetrue)

        # additive genetic variance
        eva   = 2. * np.mean ( eva_ell )
        s_eva = (K / (K-1.)) * ( 4.*np.mean(eva_ell**2) - (eva**2) )
        va    = 2. * np.mean ( va_ell )
        s_va  = (K / (K-1.)) * ( 4.*np.mean(va_ell**2) - (va**2) )

        # sample correlation coefficient
        r2 = np.nanmean (ancient_correlations**2)
        s_r2 = np.nanvar (ancient_correlations**2)

        # other estimate of rho
        rho2_trait = (np.mean (ancient_cov)**2) / (np.mean (ancient_varyhat) * np.mean (ancient_vary))


        # save point estimates to file
        #self.write_matrix (X=np.column_stack ((bias, s_bias, mse, s_mse, eva, s_eva, va, s_va, rho_tau, r2, s_r2, rho2_trait)),
        #                    outputdir=outputdir, name='samples',
        #                    header='bias,s_bias,mse,s_mse,eva,s_eva,va,s_va,rho_tau,r2,s_r2,rho2_trait')


        return np.array([time, bias, s_bias, mse, s_mse, eva, s_eva, va, s_va, rho_tau, r2, s_r2,rho2_trait])



    #def bootstrap_covariance (self, n_bootstrap, n_k, est_cov, est_var_yhat, est_var_y) :

    #    new_estimates = np.zeros (n_boostrap)
    #    for i in range (n_bootstrap) :
    #        sample_indices = np.random.randint (0, n_bootstrap, n_k)
    #        rho_i = (np.mean (est_cov[sample_indices])**2) / (np.mean (est_var_yhat[sample_indices]) * np.mean (est_var_y[sample_indices]))









    def write_matrix (self, X, outputdir, name, header=None) :
        """
        Writes a matrix to file.
        """

        fname = str(name) + '_' + str(int(self.tau)) + '.csv'
        np.savetxt (fname=os.path.join(outputdir, fname), X=X,
                    delimiter=",", header=header, comments='')


