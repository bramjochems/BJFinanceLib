# -*- coding: utf-8 -*-
from math import exp, sqrt, log
from numbers import Number
import numpy as np
import scipy.interpolate as ip
from scipy.linalg import cholesky
import BJFinanceLib.simulation.rng as Randoms
from BJFinanceLib.simulation.utils import preprocessSampleTimes, validateNumberParam

class UnivariateBlackScholesHullWhite1FGenerator:
    """
    Generates MC paths for a Hybrid Hull-White (1F) - Black-Scholes model for
    a single equity
    """    
    def __init__(self,eq_spot,eq_vol,eq_divyield,
                 rates,hw_kappa,hw_sigma,correl,sampleTimes):
        """ Constructor
        
        Arguments:
           - eq_spot: spot for the equity process at the start
           - eq_vol : annualized equity_volatility
           - divyield : dividend yield for the equity: constant
           - rates : initial yield curve. Can be a function, a number, or a
                     list like structure of tuples of time to maturity and rate
           - hw_kappa : mean reversion speed for hull white process
           - hw_sigma : volatility for hull white process
           - correl: correlation between the equity and rates process
           - sampleTimes: times at which the process is to be sampled.
        """
        # Validation of parameters
        for (param,lb) in [(eq_spot,0), (eq_vol,0),(hw_kappa,0),(hw_sigma,0)]:
            validateNumberParam(param,lb)
        validateNumberParam(correl,-1,1)
        validateNumberParam(eq_divyield)
        
        # Setting instance variables
        self.eq_spot = eq_spot
        self.eq_vol = eq_vol
        self.hw_kappa = hw_kappa
        self.rate_vol = hw_sigma
        self.correl = correl
        self.divyield = eq_divyield
        self.sampleTimes = preprocessSampleTimes(sampleTimes)
        self.timeIntervals = list(zip(self.sampleTimes[:-1],self.sampleTimes[1:]))
        self._rng = Randoms.MultidimensionalRNG(len(self.timeIntervals),2)          
        self._CholeksyUpper = cholesky(np.asarray([[1,self.correl],[self.correl,1]]))
        
        # processing input rate curve depending on what form it has (function,
        # constant, list)
        if hasattr(rates,'__call__'):
            self.rate = rates
        elif isinstance(rates,Number):
            self.rate = lambda t:rates
        else: # assume it's array/list of tuples:
            x,y = zip(*rates)
            f = ip.interp1d(x,y,fill_value="extrapolate")
            self.rate = lambda t:1*f(t)  
               
 
        # Precomputation of various quantities for efficiency reasons
        self._dt = [t-s for (s,t) in self.timeIntervals]        
        self._rate_vol_vec = [self._rate_vol(s,t) for (s,t) in self.timeIntervals] # volatility terms for time intervals
        self._mean_rev_vec = [self._mean_rev_factor(s,t) for (s,t) in self.timeIntervals] # mean reversion factor for x for time intervals
        self._eq_drift_vec = [ -self.divyield*(t-s) - 0.5*self.eq_vol**2*(t-s) for (s,t) in self.timeIntervals]  # deterministic drift part on equity component
        self._eq_vol_vec = [self._equity_vol(s,t) for (s,t) in self.timeIntervals] # weights for the vol from the equity process      
        self._phi_vec = [self._phi(t) for t in self.sampleTimes] #phi(t) for the various sample times
                 
    def _phi(self,t):
        """
        Returns phi(t) as defined by Brigo (e.g. page 884)
        """
        return self.rate(t) + (self.rate_vol*(1-exp(-self.hw_kappa*t)))**2/(2*self.hw_kappa**2)
    
    def _rate_vol(self,s,t):
        """
        Returns the standard deviation of the noise term from time s to time t
        """
        if s < t:
            return self.rate_vol*sqrt((1-exp(-2*self.hw_kappa*(t-s)))/(2*self.hw_kappa))       
        else:
            return 0
            
    def _equity_vol(self,s,t):
        """ Standard deviation of equity noise term from time s to t """
        if s < t:
            return self.eq_vol*sqrt(t-s)
        else:
            return 0.

    def _mean_rev_factor(self,s,t):
        """ mean reversion factor from time s to t """
        return exp(-self.hw_kappa*(t-s))
    
    def _getRandoms(self):
        """ Returns correlated randon normals with standard deviation of one """
        return np.dot(self._rng.getUncorrelatedNormals(),self._CholeksyUpper)
    
    def getPath(self,randomsToUse=None):
        """ return path for short rate and equity 

        randomsToUse allows specifying the random numbers to use for the paht.
        Note that if random numbers are specified as an input, no correlating
        between them is done.
        """
        if randomsToUse==None:        
            randomsToUse = self._getRandoms()
        else:
            if np.shape(randomsToUse) != (len(self.timeIntervals),2):
                raise Exception("Provided randoms are not correctly dimensioned")
        
        # Generate path for x, which is used for both equity and rate path
        x = np.zeros(len(self.sampleTimes))
        x[0] = 0
        for i in range(1,len(x)):
            x[i] = x[i-1]*self._mean_rev_vec[i-1] + self._rate_vol_vec[i-1]*randomsToUse[i-1,0]
        
        # Create rate path from x (trivial)
        r = x + self._phi_vec # short rate path
        
        # Create stock path from x
        DlnS = np.zeros(len(self.sampleTimes))
        DlnS[0] = log(self.eq_spot)        
        for i in range(1,len(DlnS)):
            DlnS[i] = (r[i]*self._dt[i-1] + # could also just do the integral of x exactly and only the phi integral numerically
                       self._eq_drift_vec[i-1] +
                       self._eq_vol_vec[i-1]*randomsToUse[i-1,1] )
        DS = np.exp(DlnS)
        S = np.cumprod(DS)       
        
        # Put r and S as two columsn next to each other and return them
        return np.column_stack([r,S])