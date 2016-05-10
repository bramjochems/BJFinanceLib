# -*- coding: utf-8 -*-
"""
Calculates risk measures (VaR, CVaR) using the delta-normal method. The delta-
normal method assumes returns are normally distributed with a given covariance
matrix.

@author: Bram Jochems
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

def _vectorify(x):
    """
    Turns an object that is supposed to contain a vector into a 1D numpy array.
    The normal np.array() function works except for nx1 dataframes where an nx1
    array is returned, so that is being taken care of as a special case
    """
    if isinstance(x,pd.DataFrame):
        return np.reshape(np.array(x),len(x))
    else:
        return np.array(x)


def _validateInput(volarray,exparray,corr_mat):
    """
    Validates the input volatities vector, exposure array and correlation
    matrix. All are assumed to have been turned into numpy arrays already.
    Does not return anything, but throws an error if dimensions are
    inconsistent.
    """
    [nr,nc] = corr_mat.shape
    if nr != nc:
        raise ValueError("Input correlation matrix isn't square")
        
    if  len(exparray) != nr:
        raise ValueError("Exposures input incorrectly sized")   
        
    if (len(volarray) != nr):
        raise ValueError("Volatility input incorrectly sized")
        
        
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

    # Make sure that all input data are np arrays rather than e.g. data frames
    # or lists.
    if isinstance(correlation,pd.DataFrame):
        corr_mat = correlation.values
    else:
        corr_mat = correlation

    [nr,nc] = corr_mat.shape
    if not len(volatilities):
        volatilities = np.ones((nr,))
    else:
        volatilities = _vectorify(volatilities)
    exposures = _vectorify(exposures)
    
    # validate inputs
    _validateInput(volatilities,exposures,corr_mat) 

    # Actual calculation. Note that the calculation for quantile and z-score
    # could be a little shorter by using the symmetry of the normal distribution
    # but I kept it like for legbility reasons since there is no significant
    # performance impact anyway.
    cov_matrix =  volatilities[:,np.newaxis]* correlation * volatilities
    var = exposures.dot(cov_matrix.dot(exposures))
    quantile = (100-np.clip(clevel,0.0,100.0))/100.0
    zscore = abs(norm.ppf(quantile))
    return zscore*np.sqrt(var)