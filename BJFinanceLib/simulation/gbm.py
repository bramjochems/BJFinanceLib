# -*- coding: utf-8 -*-

import BJFinanceLib.simulation.rng as Randoms
from numbers import Number
import numpy as np
from scipy.linalg import cholesky

def _preprocessSampleTimes(sampleTimes):
    """
    Helper function that takes sample tiime points for a monte carlo routine as
    input and 'sanitizes' it.
    """
    if isinstance(sampleTimes,Number): sampleTimes = [sampleTimes]
    return [0] + [t for t in sorted(sampleTimes) if t > 0]

def _getForwardExpDriftArray(drift,timeIntervals):
    """
    Helper function that takes a drift and timeIntervals and determines for
    each interval (t1,t2) exp(drift*(t1-t0)) which is needed for forwards in
    monte carlo simulations.
    
    Paramters:
        -drift : either a scalar of a time-varying function (implemented as)
                 a callable object. In both cases, the drift has to represent
                 the annualized continuously compounded drift
        -timeIntervals: set of periods to determine the forward drift factors
                        for. Iterable of tuples of the form
                        [(0,t1),..(ti,ti+1), (ti+1,ti+2),...(tn-1,tn)]
    """
    if hasattr(drift,'__call__'):
        return np.exp([drift(t2)*t2-drift(t1)*t1 for (t1,t2) in timeIntervals])            
    else:
        return np.exp([drift*(t2-t1) for (t1,t2) in timeIntervals])

def _getForwardVariance(volatility,timeIntervals):
    """
    Helper function that takes a volatility and timeIntervals and determines for
    each interval (t1,t2) the forward variance for that period
    
    Paramters:
        -volatility : either a scalar of a time-varying function (implemented as)
                      a callable object. In both cases, the drift has to represent
                      the annualized volatility
        -timeIntervals: set of periods to determine the forward drift factors
                        for. Iterable of tuples of the form
                        [(0,t1),..(ti,ti+1), (ti+1,ti+2),...(tn-1,tn)]
    """
    if hasattr(volatility,'__call__'):
        forwardVariance = [(volatility(t2)**2)*t2 - (volatility(t1)**2)*t1 for (t1,t2) in timeIntervals]
    else:
        v2 = volatility**2
        forwardVariance = [v2*(t2-t1) for (t1,t2) in timeIntervals] 
    return np.array(forwardVariance)

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
                            Note that this must be sane (ordered, no dupes, all
                            values positive).
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
    
        sampleTimes = _preprocessSampleTimes(sampleTimes)        
        timeIntervals = list(zip(sampleTimes[:-1],sampleTimes[1:])) # gets iterated over twice, make it a list
        self.spot = spot        
        self.sampleTimes = sampleTimes       
        self.__numberOfFuturePoints = len(sampleTimes) - 1
        
        # determine forward drifts
        forwardDrifts = _getForwardExpDriftArray(drift,timeIntervals)
        forwardVariance = _getForwardVariance(drift,timeIntervals)           
                   
        # Handle default case for rng
        if rng:
            rng =  rng
        else:
            rng = Randoms.OneDimensionalAntitheticRNG(self.__numberOfFuturePoints)           
        
        self.__rng = rng
        self.__forwardVols = np.sqrt(forwardVariance)
        self.__itoCorrectedDrifts = forwardDrifts*np.exp(-0.5*forwardVariance)            
        
    def getPath(self,randomsToUse=None):
        """
        Generates a path for a one-dimensional geometric brownian motion
        """
        # Preallocate array of length one more than the time points needed.
        # First element is zero. All other elements are the multiplicative
        # factors such that the cumulative product up to point n is the path's
        # realization at time n.
        path = np.zeros(self.__numberOfFuturePoints+1)
        path[0] = self.spot
        if randomsToUse==None:
            randomsToUse = self.__rng.getNormals()
        elif np.shape(randomsToUse) != np.shape(self.__forwardVols):
            raise Exception('Incorrectly sized random numbes provided')
        path[1:] = np.exp(randomsToUse*self.__forwardVols)*self.__itoCorrectedDrifts
        return np.cumprod(path)
        
        
class MultivariateGBMGenerator:
    """
    Path generator for geometric brownian motion for multiple underlyings where
    drift and volatility can be time-varying functions. All underlying must have
    the same currency though, no quanto corrections are done here.
    """
    @staticmethod
    def __validateParams(spots,drifts,volatilities,correlations):
        """
        Does some validation/checking on input paramters. Note that it isn't
        checked whether the correlation is a PSD matrix, because if it isn't,
        then an error will ensue during the cholesky decomposition anyway
        """
        r,c = np.shape(correlations)
        if r != c or r != len(spots) or r != len(drifts) or r != len(volatilities):
            raise Exception("Inconsistently sized dimensions")
    
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
            - correlations: correlation matrix for the underlyings. Assumed to
                            be fixed rather than potentially time varying.
            - sampleTimes : points for which values at the path are necessary.
                            Note that this must be sane (ordered, no dupes, all
                            values positive).
            - rng: Random number generator to use. Optional.
                   
        Returns: A random path. This path will be returned as a matrix with a
                 number of rows equal to 1 + the number of sample times (the
                 time t=0 is added). The number of columns equals the number
                 of underlyings
        """
        
        sampleTimes = _preprocessSampleTimes(sampleTimes)        
        timeIntervals = list(zip(sampleTimes[:-1],sampleTimes[1:])) # gets iterated over twice, make it a list
        self.__validateParams(spots,drifts,volatilities,correlations)                
        self.spots = spots  
        self.sampleTimes = sampleTimes       
        self.__numberOfFuturePoints = len(sampleTimes)-1
        self.numberOfUnderlyings = len(spots)        

        # Cholesky decomposition of correlation matrix
        self.CholeskyUpper = cholesky(correlations)
        # Determine forward drifts and variances and store them in arrays with
        # numberOfFuturePoints rows and numberOfUnderlying columns
        driftList = [_getForwardExpDriftArray(drift,timeIntervals) for drift in drifts]
        varList = [_getForwardVariance(volatility,timeIntervals) for volatility in volatilities]
        forwardDrifts = np.column_stack(driftList)        
        forwardVariance = np.column_stack(varList)
        
        # Handle default case for rng
        if rng:
            rng =  rng
        else:
            rng = Randoms.MultidimensionalRNG(self.__numberOfFuturePoints,self.numberOfUnderlyings)           
        
        self.__rng = rng
        self.__forwardVols = np.sqrt(forwardVariance)
        self.__itoCorrectedDrifts = forwardDrifts*np.exp(-0.5*forwardVariance)                

        
    def getPath(self,randomsToUse=None):
        """
        Generates a path for a one-dimensional geometric brownian motion
        
        Input:
           - randomsToUse: optional. If not provided, the rng provided will be
             used. If provided, these randoms will be used to calculate the path
        """
        # Preallocate array of length one more than the time points needed.
        # First element is zero. All other elements are the multiplicative
        # factors such that the cumulative product up to point n is the path's
        # realization at time n.
        path = np.zeros((self.__numberOfFuturePoints+1,self.numberOfUnderlyings))
        path[0,:] = self.spots
        if randomsToUse==None:
            randomsToUse = np.dot(self.__rng.getUncorrelatedNormals(),self.CholeskyUpper)
        elif np.shape(randomsToUse) != np.shape(self.__forwardVols):
            raise Exception('Incorrectly sized random numbes provided')
        path[1:,:] = np.exp(randomsToUse*self.__forwardVols)*self.__itoCorrectedDrifts
        return np.cumprod(path,axis=0)