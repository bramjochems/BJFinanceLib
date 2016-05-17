# -*- coding: utf-8 -*-

import BJFinanceLib.montecarlo.rng as Randoms
import numpy as np

class UnivariateGBMGenerator:
    """
    Path generator for single underlying geometric brownian motion
    """
    def __init__(self,spot,drift,volatility,sampleTimes,rng=None):
        """
        Constructor.

        Arguments:
            - spot: a scalar that defines the spot of the class
            - drift : either a scalar or a callable object that returns an
                      annualized drift rate when called for a time t
            - volatility : either a scalar or a callable object that returns an
                           annualized volatility when called for a time t
            - sampleTimes : points for which values at the path are necessary.
                            Must be an iterable.
            - rng: Random number generator to use. Optional, if none specified,
                   an antithetic generator based on numpy is used. On this
                   object, a call is made to a method 'getNormals', which
                   returns an array of standard normally distributed randoms.
                   
        Returns: A random path. This path will be sampled at t=0 (with value 
                 spot) followed by all times in sampleTimes. Note that
                 sampleTimes is sanitized first by removing dupes and
                 non-positive entries, as well as by sorting, so if it wasn't
                 sane to start with, the return array maybe of unexpected
                 length or ordering.
        """
        
        sampleTimes = self._preprocessSampleTimes(sampleTimes)
        # tuples of the form (0,t1), (t1,t2),..,(t(n-1),tn) for all sampleTimes        
        timeIntervals = list(zip([0] + sampleTimes[:-1],sampleTimes)) # gets iterated over twice, make it a list
        
        # determine forward drifts
        if hasattr(drift,'__call__'):
            forwardDrifts = np.exp([drift(t2)*t2-drift(t1)*t1 for (t1,t2) in timeIntervals])            
        else:
            forwardDrifts = np.exp([drift*(t2-t1) for (t1,t2) in timeIntervals])
            
        # determine forward variances
        if hasattr(volatility,'__call__'):
            forwardVariance = [(volatility(t2)**2)*t2 - (volatility(t1)**2)*t1 for (t1,t2) in timeIntervals]
        else:
            v2 = volatility**2
            forwardVariance = [v2*(t2-t1) for (t1,t2) in timeIntervals]            
        forwardVariance = np.array(forwardVariance)
        
        # Handle default case for rng
        if rng:
            rng =  rng
        else:
            rng = Randoms.OneDimensionalAntitheticRNG(self.numberOfFuturePoints)           
        
        # Set all the class attributes
        self.spot = spot        
        self.numberOfFuturePoints = len(sampleTimes) 
        self._rng = rng
        self.forwardVols = np.sqrt(forwardVariance)
        self.itoCorrectedDrifts = forwardDrifts*np.exp(-0.5*forwardVariance)            

    def _preprocessSampleTimes(self,sampleTimes):
        return [t for t in sorted(sampleTimes) if t > 0]
        
    def getPath(self):
        """
        Generates a path for a one-dimensional geometric brownian motion
        """
        # Preallocate array of length one more than the time points needed.
        # First element is zero. All other elements are the multiplicative
        # factors such that the cumulative product up to point n is the path's
        # realization at time n.
        path = np.zeros(self.numberOfFuturePoints+1)
        path[0] = self.spot
        path[1:] = np.exp(self._rng.getNormals()*self.forwardVols)*self.itoCorrectedDrifts
        return np.cumprod(path)