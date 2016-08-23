# -*- coding: utf-8 -*-


from math import exp, sqrt
from numbers import Number
import numpy as np
import scipy.interpolate as ip
import BJFinanceLib.simulation.rng as Randoms
from BJFinanceLib.simulation.utils import preprocessSampleTimes, validateNumberParam

class HullWhite1FGenerator:
    """
    Class that generates paths for a 1F Hull-White process:
        
        dr(t) = ( theta(t) - kappa r(t) )dt + sigma dW
        
    So with a constant mean-reversion speed and volatility. The theta(t) term
    is used to fit a term structure.
    
    This generator generates paths by directly numerically approximating the
    SDE, rather than decomposing r(t) in an OU process and using some analytical
    results known for it.
    """    
    
    def __init__(self,rates,kappa,sigma,sampleTimes,rng=None):
        """ Constructor
        
        Arguments:
            - rates : zero rates; can be a function, constant or list(like)
            - kappa : mean-reversion speed. Constant
            - sigma : volatility of the noise. Constant.
        """
        validateNumberParam(sigma,0)
        validateNumberParam(kappa,0)
        self.kappa = kappa
        self.sigma = sigma
        self.sampleTimes = preprocessSampleTimes(sampleTimes)
        self.timeIntervals = list(zip(sampleTimes[:-1],sampleTimes[1:]))
        
        if rng==None:
            self.__rng = Randoms.OneDimensionalAntitheticRNG(len(self.timeIntervals))  
        else:
            self.__rng = rng       
                
        if hasattr(rates,'__call__'):
            self.rate = rates
        elif isinstance(rates,Number):
            self.rate = lambda t:rates
        else: # assume it's array/list of tuples:
            x,y = zip(*rates)
            f = ip.interp1d(x,y,fill_value="extrapolate")
            self.rate = lambda t:1*f(t)         

        self._stdvec = np.array([self.stdev(s,t) for (s,t) in self.timeIntervals])
        self._weightvec = np.array([exp(-self.kappa*(t-s)) for (s,t) in self.timeIntervals])        
        self._alphavec = np.array([self.alpha(t) - 
                                   self.alpha(s)*exp(-self.kappa*(t-s)) for 
                                       (s,t) in self.timeIntervals])
        
    def stdev(self,s,t):
        """
        Returns the variance of the noise term from time s to time t
        """
        if s < t:
            return self.sigma*sqrt((1-exp(-2*self.kappa*(t-s)))/(2*self.kappa))        
        else:
            return 0
            
    def alpha(self,t):
        """
        Returns an function alpha that is used for fitting to the initial term
        structure. See the getPath function documetnation for how alpha is used
        in the monte carlo sim
        """
        return self.rate(t) + (self.sigma**2/(2*self.kappa**2))*(1-exp(-self.kappa*t))**2
                
            
    def getPath(self,randomsToUse=None):
        """
        Returns a path of the short rate in a HW 1f model

        No explicit calculation of theta(t) is used. Instead, it is used that
            
            r(t) = (r(s) - alpha(s) )exp(...) + alpha(t) + B  

        Here B normally distributed with a variance given by the variance
        function in this class. alpha(t)is given by the alpha function of this
        class.
        
        """
        if randomsToUse==None:
            randomsToUse = self.__rng.getNormals()
        elif np.shape(randomsToUse) != np.shape(self._stdvec):
            raise Exception('Incorrectly sized random numbes provided')
        
        noise_vec = self._stdvec*randomsToUse
        
        res = np.zeros(len(self.sampleTimes))
        res[0] = self.rate(0)
        for cntr in range(1,len(self.sampleTimes)):
            res[cntr] = (res[cntr-1]*self._weightvec[cntr-1] + 
                         self._alphavec[cntr-1] + noise_vec[cntr-1])          
        return res