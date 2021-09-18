import scipy
import scipy.special
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

class EqualTheory :

    def __init__(self, a, d, n, times, approx=False, d2=None) :

        # makes warnings exceptions
        #np.seterr(all='raise')

        # set values
        self.a    = float(a)
        self.d    = int(d)
        self.n    = int(n)
        self.taus = times

        # globally useful things
        self.norm  = scipy.special.beta (self.a,self.a)
        self.approx = bool(approx)

        if d2 is None :
            self.d2 = d2
        else :
            self.d2 = int(d2)


    def process (self, nanc=None, sigmaprime=None) :
        """
        """

        # set p values
        self.set_ps ()

        # check if asymmetry
        if self.d2 is not None :
            self.bias       = self.asymmetric_bias ()
            self.biasapprox = self.asymmetric_bias_approx ()
        else :
            self.bias = self.compute_bias ()

        #p3i = self.compute_p3i (dval=self.d)
        #p3i = self.compute_p3i_approx (dval=self.d)

        # compute stats
        self.mse, self.mseapprox = self.compute_mse ()
        self.eva, self.evaapprox = self.compute_vahat (nanc=nanc)
        self.va = self.compute_trueva ()
        self.compute_rho_tau (sigmaprime=sigmaprime)


    def compute_trueva (self) :
        return 2. * (self.a / (2.*(2.*self.a + 1.)))


    def set_ps (self) :
        """
        Set P1 - P4 values. But, first test to see if should use exact or approx.
        values.
        """

        testbinom = scipy.special.binom (2*self.n,self.d)
        if np.isfinite (testbinom) and not self.approx :
            self.set_ps_exact ()
        else :
            self.set_ps_approx ()


    def set_ps_exact (self) :
        self.p3 = self.compute_p3 ()
        self.p1 = self.compute_p1 ()
        self.p2 = self.compute_p2 ()
        self.p4 = self.compute_p4 ()


    def set_ps_approx (self) :
        self.p3 = self.compute_p3 ()
        self.p1 = self.compute_p1_approx ()
        self.p2 = self.compute_p2_approx ()
        self.p4 = self.compute_p4_approx ()


    def compute_p1 (self) :
        """
        Computes P_1^d.
        """

        cumsum = 0.
        for i in range (0,self.d+1) :
            mean  = ((i - self.n)/self.n)**2
            beta  = (scipy.special.binom (2*self.n,i) / self.norm )
            beta *= scipy.special.beta (self.a+float(i), self.a + 2.*self.n - float(i))
            cumsum += (mean * beta)

        return cumsum


    def compute_p2 (self) :
        """
        Computes P_2^d
        """

        cumsum = 0.
        for i in range (0,self.d+1) :
            mean  = ((i - self.n)**2)  / (self.n*(self.a+self.n))
            beta  = (scipy.special.binom (2*self.n,i) / self.norm )
            beta *= scipy.special.beta (self.a+float(i), self.a + 2.*self.n - float(i))

            cumsum += (mean * beta)

        return cumsum


    def compute_p3 (self, dval=None) :
        """
        Computes P_3^d.
        """

        if dval is None :
            dval = self.d

        if dval < self.n :
            p3 = scipy.stats.betabinom.cdf (dval, 2.*self.n, self.a, self.a)
        else :
            p3 = 0.5

        return p3


    def compute_p4 (self) :
        """
        Computes P_4^d.
        """

        cumsum = 0.
        for i in range (0,self.d+1) :
            mean  = (2.*self.a + 1.)*i*(i-2.*self.n) + self.a*self.n*(2.*self.n-1.)
            mean /= ((2.*self.a + 2.*self.n +1.)*(self.a+self.n))
            beta  = (scipy.special.binom (2*self.n,i) / self.norm )
            beta *= scipy.special.beta (self.a+float(i), self.a + 2.*self.n - float(i))

            cumsum += (mean * beta)

        return cumsum


    def compute_p1_approx (self) :

        x = self.d / (2.*self.n)
        inc1 = scipy.special.betainc (self.a+1,self.a,x) * scipy.special.beta  (self.a+1,self.a) / self.norm
        inc2 = scipy.special.betainc (self.a+2,self.a,x) * scipy.special.beta (self.a+2,self.a) / self.norm

        return 4.*inc2 - 4.*inc1 + self.p3


    def compute_p2_approx (self) :

        return (self.n / (self.a + self.n)) * self.p1


    def compute_p4_approx (self) :

        x = self.d / (2.*self.n)
        inc1 = scipy.special.betainc (self.a+1,self.a,x) * scipy.special.beta  (self.a+1,self.a) / self.norm
        inc2 = scipy.special.betainc (self.a+2,self.a,x) * scipy.special.beta (self.a+2,self.a) / self.norm

        denom = (2.*self.a + 2.*self.n + 1.) * (self.a + self.n)
        term1 = (self.a * self.n * (2.*self.n - 1.)) * self.p3
        term2 = (2.*self.a + 1.) * (4.*self.n**2) * (inc2 - inc1)

        return (term1 + term2) / denom


    def compute_p3i_approx (self,dval) :
        """
        Approximates P_3 i using a Beta distribution.
        """

        x = dval / (2.*self.n)
        inc1 = scipy.special.betainc (self.a+1,self.a,x) * scipy.special.beta  (self.a+1,self.a) / self.norm

        #print('approx p3i: ' + str(inc1 * 2. * self.n))

        return (inc1 * 2. * self.n)


    def compute_p3i (self,dval) :
        """
        Computes the first moment of a truncated Beta-Binomial.
        """

        testbinom = scipy.special.binom (2*self.n,dval)
        if np.isfinite (testbinom) and not self.approx and dval <= 1000:

            cumsum = 0.
            for i in range (0,dval+1) :
                beta    = (scipy.special.binom (2*self.n,i) / self.norm )
                beta   *= scipy.special.beta (self.a+float(i), self.a + 2.*self.n - float(i))
                cumsum += (beta * i)

        else :
            cumsum = self.compute_p3i_approx (dval=dval)

        return cumsum


    def compute_rho_tau (self,sigmaprime=None) :

        numerator, approx   = self.compute_vahat (nanc=None)
        denominator = self.a / (2.*self.a + 1.)

        if sigmaprime is not None :
            denominator += sigmaprime

        self.rho_tau = (numerator) / denominator


    def compute_mse (self) :
        """
        Computes the mean-squared error.
        """

        mse  = ((self.a+1.) / (2.*self.a + 1.)) * self.p3
        mse += ((1. / (2.*self.a + 1.)) * np.exp (-(2.*self.a + 1.)*self.taus) * self.p4)
        mse -= (2. * np.exp (-self.a*self.taus) * self.p2)
        mse += self.p1

        mseapprox = 2.*mse[0] + (2.*self.a*self.p3*self.taus)

        return 2.*mse, mseapprox


    def compute_vahat (self, nanc=None) :
        """
        Compute expected hat Va.
        """

        t1 = (self.a / (2.*(2.*self.a + 1.))) * (1. - 2.*self.p3)
        t2 = np.exp (-(2.*self.a + 1.)*self.taus) * (1. / (2.*self.a+1.)) * self.p4

        v0 = t1 + (1. / (2.*self.a+1.)) * self.p4
        approx  = v0 - (self.a*self.p3*self.taus)

        if nanc is not None :
            ancestralmult = 2.*((2.*nanc-1.)/(2.*nanc))
        else :
            ancestralmult = 2.

        approx[approx < 0] = np.NaN

        return (t1 + t2)*ancestralmult, approx*ancestralmult


    def compute_bias (self) :

        return np.zeros (len(self.taus))


    def asymmetric_bias (self) :
        """
        Compute bias for asymmetric values of d.
        """

        p3d2  = self.compute_p3 (dval=self.d2)
        p3d1i = self.compute_p3i (dval=self.d)
        p3d2i = self.compute_p3i (dval=self.d2)

        mult = (1./self.n) - np.exp (-self.a*self.taus) * (1. / (self.a+self.n))

        return mult * ( self.n * (p3d2 - self.p3) + p3d1i - p3d2i)


    def asymmetric_bias_approx (self) :
        """
        Compute approx bias for asymmetric values of d.
        """

        p3d2  = self.compute_p3 (dval=self.d2)

        return self.a * self.taus * (p3d2 - self.p3)


    def test (self) :

        self.compute_p1 ()
        self.compute_p2 ()
        self.compute_p3 ()
        self.compute_p3i ()
        self.compute_p4 ()
