# -*- coding: utf-8 -*-
"""
Calculates risk measures (VaR, CVaR) using the delta-normal method. The delta-
normal method assumes returns are normally distributed with a given covariance
matrix.

@author: Bram Jochems
"""
import numpy as np
from scipy.stats import norm

def VaR(clevel, exposures,correlation,volatilities=None):
    """
    Calculates VaR using the delta-normal method.
    
    Arguments:
       - clevel: the confidence level for the VaR. Between 0 and 100
       - exposures: vector of exposures to the risk factors
       - correlation: correlations between the risk factors. Note that if the
         argument volatilities isn't passed in, volatilities are all assumed to
         equal one; combined with no argument checking on the correlation input,
         this means that a covariance matrix can be provided as input as well.
       - volatilities: input vector of volatilites for the various factors.
         Optional, default=None. If None is provided as a value, then a vector of
         ones is used, which means that the correlation matrix can be a covariance
         matrix.
         
    Returns: a VaR estimate for the given exposure assuming normally distributed
             risk factors with given correlation structure and volatilities.
    """
    quantile = (100.0 - np.clip(clevel,0.0,100.0))/100.0
    zscore = abs(norm.ppf(quantile)) #combined the above two transformations
                                     #are mostly superfluous (except the
                                     #clipping, but this way is consistent with
                                     #the other VaR functions)
    [nr,nc] = correlation.shape
    if nr != nc or len(exposures) != nc:
        raise ValueError("Arguments are inconsistently sized")
    exposures = np.array(exposures)
    if not volatilities:
        volatilities = np.ones((nr,1))
    else:
        volatilities = np.array(volatilities)
    cov_matrix =  volatilities* correlation * volatilities[:,np.newaxis]
    var = exposures.dot(cov_matrix.dot(exposures))
    return zscore*np.sqrt(var)