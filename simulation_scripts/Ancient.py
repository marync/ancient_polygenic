import numpy as np
import scipy.special
import scipy.stats


class Ancient :
    """
    """

    def sample_ancient_genotype ( self, ps ) :

        return np.random.binomial(n=self.ploidy, p=ps) - 1.


    def compute_ancient_polygenic( self, x, betahats, intercept ) :
        """
        Compute polygenic score.
        """

        return np.inner(x, betahats) + intercept


    def compute_ancient_phenotype ( self, x ) :
        """
        Sample ancient phenotype.
        """

        enviro   = np.random.normal(loc=0, scale=self.sigma)
        genetics = np.sum(x) * self.beta

        return genetics + enviro


    def sample_ancient ( self, frequencies ) :
        """
        """

        genotype  = self.sample_ancient_genotype( ps=frequencies )
        #polyscore = self.compute_ancient_polygenic( betahats=betahats,
        #                                            x=genotype,
        #                                            intercept=intercept )
        phenotype = self.compute_ancient_phenotype( x=genotype )

        return genotype, phenotype


    def sample_na_ancients (self, frequencies, na, betas, intercept) :

        # sample ancient genotypes
        genotypes = np.zeros ((na, len(frequencies)))
        for i in range (na) :
            genotypes[i,:] = self.sample_ancient_genotype (ps=frequencies) + 1.

        ancient_scores = self.compute_ancient_polygenic (x=genotypes, 
                                                              betahats=betas,
                                                              intercept=intercept)

        ancient_phens  = np.sum(genotypes, axis=1) * self.beta
        ancient_phens += np.random.normal(loc=0, scale=np.sqrt(self.sigma), size=na)

        return ancient_scores, ancient_phens



    def estimate_ancient_correlation (self, yhat, y) : 

        return np.corrcoef (yhat, y)[0,1]


    def estimate_rho_tau (self, yhat, y) :

        covariance    = np.cov (yhat, y)[0,1]
        variance_haty = np.var (yhat)
        variance_y    = np.var (y)

        return covariance, variance_haty, variance_y




    def estimate_ancient_va ( self, frequencies, betahats, nancient ) :
        """
        Compute the estimated ancient V_A.
        """

        #print( 'mean beta hat: ' + str(np.mean(betahats)) )

        nancient = int(nancient)

        # sample ancient genotypes
        genotypes = np.zeros ((nancient, len(frequencies)))
        for i in range (nancient) :
            genotypes[i,:] = self.sample_ancient_genotype (ps=frequencies) + 1.

        # compute the allele frequencies
        zhat = (np.sum (genotypes, axis=0)) / (2.*float(nancient))
        #print('zhat: ' + str(np.unique(zhat)))

        # estimate va
        vahat = 2. * np.sum ((betahats**2)*zhat*(1.-zhat))
        #print('vahat: ' + str(vahat))

        return vahat


    def true_va ( self, frequencies ) :
        """
        Compute the true additive genetic variance in the ancient population.
        """

        va = np.sum ( frequencies * (1. - frequencies) )

        return 2. * (self.beta**2) * va


