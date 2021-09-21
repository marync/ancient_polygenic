import matplotlib.pyplot as plt
import numpy as np
import time

class Gwas :
    """
    Functions for simulating the GWA study.
    """

    def sample_genotypes_and_counts ( self, ps ) :
        """
        """


        #genos  = np.random.binomial(n=self.ploidy, p=ps, size=(self.n,1,self.L))
        counts  = np.random.binomial(n=self.ploidy*self.n, p=ps)

        #genos  = genos.reshape ((self.n, self.L))
        #counts = np.sum (genos, axis=0)
        xbar = (counts - self.n) / self.n

        return xbar, counts


    def sample_phenotypes ( self, X ) :
        """
        """

        enviro   = np.random.normal(loc=0, scale=self.sigma, size=self.n)
        genetics = np.sum(X, axis=1) * self.beta

        return genetics + enviro


    def sample_ybar ( self, Xbars ) :
        """
        """

        enviro   = np.random.normal(loc=0, scale=self.sigma, size=self.n)
        genetics = np.sum(Xbars) * self.beta

        return genetics + np.mean(enviro)



    def estimate_beta( self, counts, lower=None, upper=None ) :
        """
        Simple threshold model.
        """

        betas = np.zeros( int(self.L) )

        if lower is None :
            lower = self.epsilon
        if upper is None :
            upper = (2.*float(self.n) - self.epsilon)

        #print('lower: ' +str(lower) + ', upper: ' + str(upper))

        betas[np.where(np.logical_and(counts>lower, counts<upper))] = self.beta
        #print(counts[np.where (betas == self.beta)])
        #print ()
        #print (counts[np.where (betas == 0)])

        #betas[np.where(np.logical_and(counts>=lower, counts<=upper))] = self.beta

        #loops = time.time ()
        #betas2 = np.zeros ( int (self.L) )
        #for l in range ( int(self.L) ) :
        #    d = counts[l]
        #    if (d > self.epsilon) and (d < (2.*float(self.n) - self.epsilon)) :
        #        betas2[l] = self.beta
        #loopf = time.time ()
        #print('loop: ' + str(loopf - loops))

        #loops = time.time ()
        #loopf = time.time ()
        #print('array: ' + str(loopf - loops))
        #print('are they equal: ' + str(np.sum(betas2)) + ', ' + str(np.sum(betas)))


        return betas


    def estimate_chat_ybar ( self, Xbars, Ybar, betahats ) :
        """
        MLE of the intercept term.
        """

        return ( Ybar - np.inner(Xbars, betahats) )


    def estimate_chat( self, X, Y, betahats ) :
        """
        MLE of the intercept term.
        """

        bary = np.mean(Y)
        barx = np.mean(X, axis=0)
        chat = bary - np.inner(barx, betahats)

        return chat


    def conduct_gwas (self, frequencies, lower=None, upper=None) :
        """
        """

        #check = np.random.uniform (0,1,5)
        #print('check: ' + str(check))
        # gwas
        #genotypes, counts = self.sample_genotypes_and_counts ( ps=frequencies )
        #phenotypes        = self.sample_phenotypes ( X=genotypes )
        #chat              = self.estimate_chat ( X=genotypes, Y=phenotypes, betahats=betahats)

        xbars, counts = self.sample_genotypes_and_counts ( ps=frequencies )
        ybar = self.sample_ybar (xbars)
        betahats          = self.estimate_beta ( counts=counts, lower=lower, upper=upper )
        chat              = self.estimate_chat_ybar ( Xbars=xbars, Ybar=ybar, betahats=betahats)

        return betahats, chat, xbars
