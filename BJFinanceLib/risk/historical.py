# -*- coding: utf-8 -*-
"""
Various methods for estimating risk measures based on a history of returns,
such as historcal VaR and expected shortfall.

@author: Bram Jochems
"""

import numpy as np

def VaR(pnldata,clevel):
    """
    Calculates the Value-at-Risk for historical pnl data. For a vector r of
    returns, the a% VaR is defined as -Q(r,100-a) where Q is the quantile
    function. Note that by this definition, a VaR number representing a loss
    is returned as a positive number.
    
    Arguments:
        - pnldata : this can be either any iterable containing numeric values
                    or a dataframe or numpy matrix. In those cases, each
                    column is interpreted as a seperate input for which a VaR
                    figure is returned.
        - clelvel: The "confidence level" for which to calculate the VaR number.
                   Confidence levels are expressed as a number between 0 an 100
                   instead of between zero and one
                   
    Returns: either a scalar (if pnldata is one-dimensional) or a numpy vector
             (if pnldata is two-dimensional) where the values represent the
             value at risk number(s) associated with the input
    """
    quantile = 100 - np.clip(clevel,0,100)
    return -np.percentile(pnldata,quantile,axis=0)
    
def CVaR(pnldata,clevel):
    """
    Calculates the Conditional Value-at-Risk (also called Expected Shortfall,
    Average Value-at-Risk or Expected Tail Loss) for historical pnl data. For
    a vector r of returns, the a% CVaR is defined as
    
        -E[r | r < Q(r,100-a)]
        
    where Q is the quantile function. In other words, it is the expected level
    of pnls that are below the associated VaR level. Note by this definition,
    an average loss is represented as a positive number.
    
    Arguments:
        - pnldata : this can be either any iterable containing numeric values
                    or a dataframe or numpy matrix. In those cases, each
                    column is interpreted as a seperate input for which a VaR
                    figure is returned.
        - clelvel: The "confidence level" for which to calculate the VaR number.
                   Confidence levels are expressed as a number between 0 an 100
                   instead of between zero and one
                   
    Returns: either a scalar (if pnldata is one-dimensional) or a numpy vector
             (if pnldata is two-dimensional) where the values represent the
             value at risk number(s) associated with the input
    """
    res = -np.apply_along_axis(lambda x:np.average(x[x<=-VaR(x,clevel)]),
                               0,pnldata)    
    # If pnldata is a one-dimensional structure, the output is in the form of
    # an ndarray(..) with .. a scalar and the form of this is (), but it isn't
    # recognized as a scalar. To get the output to the correct format, an
    # explicit shape test and ocnversion is done for this. I'm sure this can
    # be done more efficiently, but I don't know how.
    if np.shape(res) == ():
        return res.sum()
    else:
        return res
    
    
def ES(pnldata,clevel):
    """
    Calculates Expected Shortfall by calling the CVaR function. This function
    is only here for convenient naming access and does not provide additional
    logic. See the CVaR function for documentation.
    """
    return CVaR(pnldata,clevel)
    
def AVaR(pnldata,clevel):
    """
    Calculates Average Value-at-Risk by calling the CVaR function. This
    function is only here for convenient naming access and does not provide
    additional logic. See the CVaR function for documentation.
    """
    return CVaR(pnldata,clevel)
    
    
def ETL(pnldata,clevel):
    """
    Calculates Expected Tail Loss by calling the CVaR function. This
    function is only here for convenient naming access and does not provide
    additional logic. See the CVaR function for documentation.
    """
    return CVaR(pnldata,clevel)
    
    