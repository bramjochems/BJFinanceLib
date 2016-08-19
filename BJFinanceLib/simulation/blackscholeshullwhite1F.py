# -*- coding: utf-8 -*-

def _preprocessSampleTimes(sampleTimes):
    """
    Helper function that takes sample tiime points for a monte carlo routine as
    input and 'sanitizes' it.
    """
    if isinstance(sampleTimes,Number): sampleTimes = [sampleTimes]
    return [0] + [t for t in sorted(sampleTimes) if t > 0]

class BlackScholesHullWhite1FGenerator:
    """
    Generates MC paths for a Hybrid Hull-White (1F) - Black-Scholes model
    """    

    
    def __init__(self,
                 spot,drift,volatility,sampleTimes,
                 rng=None):
        pass
        self._rng = pass
        self._gbm_generator = pass
        self._hw_generator = pass    
    
    def getPath(self,randomsToUse=None):
        # Generate correlated randoms for the HW and BS processes
        # Project the rate and equities forward
        # return all in one array        
        
        pass

