import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import copy as cp

#from Onset import *


class Allele : #(Onset) :
    """
    Functions for simulating allele frequency trajectories.
    """


    def sample_initial_frequencies_neutral ( self ) :
        """
        """

        frequencies = np.random.beta(a=self.a, b=self.b, size=int(self.L))
        samplefreqs = self.sample_frequency_neutral (p=frequencies)

        return samplefreqs


    def sample_frequency_neutral ( self, p ) :
        """
        Generate a binomial sample under the recurrent mutation model.
        p: (float) Allele frequency.
        """

        phi = ( (1.-p) * self.mu ) + ( p * (1.-self.nu) )

        return self.sample_counts (phis=phi)


    def sample_frequency_selection( self, p ) :
        """
        Generate a binomial sample under the recurrent mutation model.
        p: (float) Allele frequency.
        """

        # mean fitness
        wbar  = 1. + (2. * p * self.s)

        # compute phi
        num  = ( (1.-p)**2 * self.mu )
        num += p * (1.-p) * (1.+self.s) * (1.-self.nu)
        num += p * (1.-p) * (1.+self.s) * (self.mu)
        num += (p**2) * (1.+2.*self.s) * (1. - self.nu)
        phi = (num / wbar)

        return self.sample_counts (phis=phi)


    def sample_counts( self, phis ) :
        count = np.random.binomial ( self.N, phis )

        return (count / self.N)


    def evolve_neutral ( self, frequencies ) :
        """
        Evolve allele frequencies forward in time.
        """

        for i in range(0, int(self.tau)) :
            frequencies = self.sample_frequency_neutral ( p=frequencies )

        return frequencies


    def sample_initial_frequencies_onset ( self ) :
        """
        Sample allele frequencies in onset scenario.
        """


        frequencies = self.sample_initial_frequencies_neutral ()

        if int(self.tau) <= int(self.onset) :
            for i in range (0, 50) :
                frequencies = self.sample_frequency_neutral ( frequencies )

            for i in range (int(self.onset) - int(self.tau)) :
                frequencies = self.sample_frequency_selection ( frequencies )

        return frequencies


    def evolve_selection_onset ( self, frequencies ) :
        """
        Evolve allele frequencies forward in time with selection.
        """

        if int(self.tau) <= int(self.onset) :
            for i in range(int(self.tau)) :
                frequencies = self.sample_frequency_selection ( frequencies )

        else :
            for i in range (int(self.tau) - int(self.onset)) :
                frequencies = self.sample_frequency_neutral ( p=frequencies )

            for i in range (int(self.onset)) :
                frequencies = self.sample_frequency_selection ( p=frequencies )


        return frequencies


    def simulate_alleles ( self, seed=None ) :
        """
        See below for arguments.
        """

        if seed is not None :
            np.random.seed ( seed )

        # initial stuff
        if self.s is None or self.s == 0 :
            ancientFreqs = self.sample_initial_frequencies_neutral ()
            contempFreqs = self.evolve_neutral ( frequencies=ancientFreqs )

        elif self.s is not None and self.onset is not None :
            ancientFreqs = self.sample_initial_frequencies_onset ()
            contempFreqs = self.evolve_selection_onset ( frequencies=ancientFreqs )

        else :
            print('Need to provide onset time if selection is non-zero')
            sys.exit ()

        return ancientFreqs, contempFreqs


