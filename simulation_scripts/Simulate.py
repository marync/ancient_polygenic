import os, sys
import numpy as np
import scipy.special
import scipy.stats
import copy as cp

from Allele import *  # Module for evolving allele frequencies

class Simulate (Allele) :

    def simulate (self, seeds=None, outputdir='.', nsim=1) :

        # execute simulations
        ancient, contemp = self.run_simulation (nsim=nsim, seeds=seeds)

        # write to files
        self.write_matrix (X=ancient, outputdir=outputdir, name='ancient')
        self.write_matrix (X=contemp, outputdir=outputdir, name='contemp')


    def run_simulation (self, nsim, seeds) :

        seed = None

        # matrices to store allele frequencies
        ancientMat = np.zeros ((self.L, nsim))
        contempMat = np.zeros ((self.L, nsim))
        for i in range(nsim) :

            # just in case
            ancientFreqs = None
            contempFreqs = None

            # simulate allele frequencies
            if seeds is not None :
                seed = seeds[i]

            ancientFreqs, contempFreqs = self.simulate_alleles (seed=seed)

            # store in two matrices
            ancientMat[:,i] = cp.deepcopy (ancientFreqs)
            contempMat[:,i] = cp.deepcopy (contempFreqs)

        return ancientMat, contempMat


    def write_matrix (self, X, outputdir, name) :
        """
        Writes a matrix to file.
        """

        fname = str(name) + '_' + str(int(self.tau)) + '.csv'
        np.savetxt (fname=os.path.join(outputdir, fname), X=X, delimiter=',',fmt='%1.5f')



    def __init__( self,
                  N=10.**4,
                  mu=10**(-4),
                  nu=10**(-4),
                  ploidy=2,
                  L=1,
                  effectSize=1.,
                  gwasSize=1e3, epsilon=0., offset=0.,
                  tau=0.,
                  sigma=0,
                  s=None, onset=None,
                  rseed=None,
                  verbose=True ) :

        # set variables
        self.N            = float(ploidy) * float(N)
        self.gwasSize     = float(ploidy) * float(gwasSize)
        self.epsilon      = int(epsilon)
        self.beta         = float(effectSize)
        self.mu           = mu
        self.nu           = nu
        self.L            = L
        self.ploidy       = int(ploidy)
        self.offset       = int(offset)
        self.tau          = float(tau)
        self.s            = s
        self.onset        = onset

        # mutation parameters
        self.a = 2. * self.N * mu
        self.b = 2. * self.N * nu

        if rseed is not None :
            self.rseed = int(rseed)

        if self.s is not None :
            self.s = float(s)
            self.onset = int(onset)

        if verbose :
            self.print_arguments ()


    def print_arguments(self) :

        # print simulation parameters
        out  = "Population size: " + str(self.N) + "\n"
        out += "GWAS population size: " + str(self.gwasSize) + "\n"
        out += "Epsilon: " + str(self.epsilon) + "\n"
        out += "Tau: " + str(self.tau) + "\n"
        out += "Effect size: " + str(self.beta) + "\n"
        out += "Forward mutation rate: " + str(self.mu) + "\n"
        out += "Backward mutation rate: " + str(self.nu) + "\n"
        out += "Number of sites: " + str(self.L) + "\n"
        out += "Selection coefficient: " + str(self.s) + "\n"
        out += "Onset: " + str(self.onset) + "\n"
        out += "Ploidy is: " + str(self.ploidy) + "\n"

        print(out)

