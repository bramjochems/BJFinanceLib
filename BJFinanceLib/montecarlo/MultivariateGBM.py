# -*- coding: utf-8 -*-


from numbers import Number
import BJFinanceLib.montecarlo.rng as Randoms
import numpy as np
from scipy.linalg import cholesky

class MultivariateGBMGenerator:
    """
    Path generator for geometric brownian motion for multiple underlyings
    """
    @staticmethod
    def __validateParams(spots,drifts,volatilities,correlations):
        r,c = np.shape(correlations)
        if r != c or r != len(spots) or r != len(drifts) or r != len(volatilities):
            raise Exception("Inconsistently sized dimensions")
        # TODO: check if correlations is PSD
    
    def __init__(self,spots,drifts,volatilities,correlations,sampleTimes,rng=None):
        """
        Constructor.

        Arguments:
            - spots: a list of spots for the various underlyings
            - drifts : a list of drifts for each underlyings. Each drift can be
                       either a scalar or a calalble function that returns the
                       annualized drift for an underlying
            - volatilities : a list where each element is either a scalar or a
                             callable object that returns an annualized
                             volatility when called for a time t
            - correlations: correlation matrix for the underlyings
            - sampleTimes : points for which values at the path are necessary.
                            Note that this must be sane (ordered, no dupes, all
                            values positive).
            - rng: Random number generator to use. Optional.
                   
        Returns: A random path. This path will be returned as a matrix with a
                 number of rows equal to 1 + the number of sample times (the
                 time t=0 is added). The number of columns equals the number
                 of underlyings
        """
        
        
        if isinstance(sampleTimes,Number): sampleTimes = [sampleTimes]
        sampleTimes = self._preprocessSampleTimes(sampleTimes)        
        timeIntervals = list(zip([0] + sampleTimes[:-1],sampleTimes)) # gets iterated over twice, make it a list
        self.__validateParams(spots,drifts,volatilities,correlations)                
        self.spots = spots  
        self.sampleTimes = [0] + sampleTimes       
        self.numberOfFuturePoints = len(sampleTimes) 
        self.numberOfUnderlyings = len(spots)        

        # Cholesky decomposition of correlation matrix
        self.CholeskyUpper = cholesky(correlations)
        # Determine forward drifts and store them in an array that has
        # numberOfFuturePoints rows and numberOfUnderlying columns
        driftList = []
        for drift in drifts:
            # determine forward drifts
            if hasattr(drift,'__call__'):
                driftList.append(np.exp([drift(t2)*t2-drift(t1)*t1 for (t1,t2) in timeIntervals]))    
            else:
                driftList.append(np.exp([drift*(t2-t1) for (t1,t2) in timeIntervals]))
        forwardDrifts = np.column_stack(driftList)
           
        # determine forward variances and store them in an array that has
        # numberOfFuturePoints rows and numberOfUnderlying columns
        varList = []
        for volatility in volatilities:
            if hasattr(volatility,'__call__'):
                forwardVariance = [(volatility(t2)**2)*t2 - (volatility(t1)**2)*t1 for (t1,t2) in timeIntervals]
            else:
                v2 = volatility**2
                forwardVariance = [v2*(t2-t1) for (t1,t2) in timeIntervals]            
            varList.append(np.array(forwardVariance))
        forwardVariance = np.column_stack(varList)
        
        # Handle default case for rng
        if rng:
            rng =  rng
        else:
            rng = Randoms.MultidimensionalRNG(self.numberOfFuturePoints,self.numberOfUnderlyings)           
        
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
        path = np.zeros((self.numberOfFuturePoints+1,self.numberOfUnderlyings))
        path[0,:] = self.spots
        path[1:,:] = np.exp(np.dot(self._rng.getUncorrelatedNormals(),self.CholeskyUpper)*self.forwardVols)*self.itoCorrectedDrifts
        return np.cumprod(path,axis=0)
