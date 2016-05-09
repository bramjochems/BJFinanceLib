# -*- coding: utf-8 -*-
"""
Various methods for estimating the volatility of a financial time series

@author: Bram Jochems
"""

import numpy as np
import pandas as pd
import scipy.special as ss

def population_stdev_correction(samplesize):
    """
    Calculates the multiplicative adjustment that must be made to a sample
    standard deviation to estimate an unbiased population standard deviation
    for a normal distribution. The normal standard deviation estimator is biased
    because the estimator for variance is unbiased
    
    Arguments:
    -----
    samplesize : integer representing the length of the window 
    
    Output
    ------
    For samplesizes larger than 1, a float that needs be multiplied with sample
    standard deviation to retrieve the population standard deviation"""

    if samplesize > 1:
        helper = 0.5*samplesize
        return np.sqrt(helper)*ss.gamma(helper-0.5)/ss.gamma(helper)
    else:
        raise ValueError('Samplesize <= 1 encountered in population_stdev_correction')
        
def estimator_Parkinson(hilo_df,colname_low='lo',colname_high='hi',annfactor=np.sqrt(252)):
    """ Estimates volatility using hi and lo prices
    
    Input
    ----
    hilo_df: A dataframe in which low and high prices for each trading day are
             present
    colname_low : the name of the column in the dataframe with the daily lows.
                  Optional, default = 'lo'
    colname_high : the name of the column in the dataframe with the daily highs.
                  Optional, default = 'hi'            

    annfactor : annualizationfactor. Optional, default= sqrt(252)

    Output
    ------
    The parkison volatility estimator. Please note that the Parkison estimator
    systematically underestimates volatility and cannot deal well with trends
    and jumps.
    """
    hilo_returns = np.log(hilo_df[colname_high]/hilo_df[colname_low])
    return max(0,annfactor)*np.sqrt(np.mean(hilo_returns*hilo_returns)/(4*np.log(2)))

def estimator_GarmanKlass(ohlc_df,
                          colname_open='open',
                          colname_low='lo',
                          colname_high='hi',
                          colname_close='close',
                          annfactor=np.sqrt(252)):
    """ Estimates volatility using ohlc data using the method from
    Garman and Klass.
    
    Input
    ----
    hiloclose_df: A dataframe in which low, high and close prices for each
                  trading day are present
    colname_open: the name of the column in the dataframe with the daily open.
                  Optional, default = 'open'
    colname_low : the name of the column in the dataframe with the daily lows.
                  Optional, default = 'lo'
    colname_high : the name of the column in the dataframe with the daily highs.
                  Optional, default = 'hi'
    colname_close: the noame of the column in the dataframe with the daily
                   close. Optional, default='close'
    annfactor : annualizationfactor. Optional, default= sqrt(252)

    Output
    ------
    The Garman-Klass volatility estimator. Note that this is quite biased
    estimator
    """    
    hl = np.log(ohlc_df[colname_high]/ohlc_df[colname_low])
    cl = np.log(ohlc_df[colname_close]/ohlc_df[colname_open])
    return max(0,annfactor)*np.sqrt(np.mean(0.5*hl*hl-(2*np.log(2)-1)*cl*cl))
                   
def estimator_RogersSatchellYoon(ohlc_df,
                                  colname_open='open',
                                  colname_low='lo',
                                  colname_high='hi',
                                  colname_close='close',
                                  annfactor=np.sqrt(252)):
    """ Estimates volatility using ohcl data using the method from Rogers,
    Satchell and Yoon
    
    Input
    ----
    hiloclose_df: A dataframe in which low, high and close prices for each
                  trading day are present
    colname_open: the name of the column in the dataframe with the daily open.
                  Optional, default = 'open'
    colname_low : the name of the column in the dataframe with the daily lows.
                  Optional, default = 'lo'
    colname_high : the name of the column in the dataframe with the daily highs.
                  Optional, default = 'hi'
    colname_close: the noame of the column in the dataframe with the daily
                   close. Optional, default='close'
    annfactor : annualizationfactor. Optional, default= sqrt(252)

    Output
    ------
    The Rogers-Satchell-Yoon volatility estimator.
    """    
    hc = np.log(ohlc_df[colname_high]/ohlc_df[colname_close])
    ho = np.log(ohlc_df[colname_high]/ohlc_df[colname_open])
    lc = np.log(ohlc_df[colname_low]/ohlc_df[colname_close])
    lo = np.log(ohlc_df[colname_low]/ohlc_df[colname_open])
    return max(0,annfactor)*np.sqrt(np.mean(hc*ho+lc*lo))    
    
def estimator_YangZhang(ohlc_df,
                        colname_open='open',
                        colname_low='lo',
                        colname_high='hi',
                        colname_close='close',
                        annfactor= np.sqrt(252)):
    """ Estimates volatility using ohcl data using the method from Yang & Zhang
    
    Input
    ----
    hiloclose_df: A dataframe in which low, high and close prices for each
                  trading day are present
    colname_open: the name of the column in the dataframe with the daily open.
                  Optional, default = 'open'
    colname_low : the name of the column in the dataframe with the daily lows.
                  Optional, default = 'lo'
    colname_high : the name of the column in the dataframe with the daily highs.
                  Optional, default = 'hi'
    colname_close: the noame of the column in the dataframe with the daily
                   close. Optional, default='close'
    annfactor : annualizationfactor. Optional, default= sqrt(252)

    Output
    ------
    The Yang-Zhang volatility estimator.
    """
    N = len(ohlc_df)
    k = 0.34/(1+(N+1)/(N-1))    
    so = np.log(ohlc_df[colname_open]/ohlc_df[colname_close].shift())
    so = np.sum(so*so)
    sc = np.log(ohlc_df[colname_close]/ohlc_df[colname_open].shift())
    sc = np.sum(sc*sc)
    hc = np.log(ohlc_df[colname_high]/ohlc_df[colname_close])
    ho = np.log(ohlc_df[colname_high]/ohlc_df[colname_open])
    lc = np.log(ohlc_df[colname_low]/ohlc_df[colname_close])
    lo = np.log(ohlc_df[colname_low]/ohlc_df[colname_open])
    sr = np.sum(hc*ho+lc*lo)
    return max(0,annfactor)*np.sqrt((so+k*sc+(1-k)*sr)/(N-1.0))
        
def volatilitycone(data,
                   estimator,
                   windows,
                   percentiles=[0.10,0.25,0.50,0.75,0.90]):
                   
    """
    Calculates volatility cones (volatility estimates for different) tenors
    for different percentile levels. Does so by calculating returns from
    overlapping  windows, but adjusting for the bias due to autocorrelation as
    per Hodgins & Tompkins.
    
    Inputs
    ------
    data : data to calculate the volatility for as a pandas dataframe
    estimator : function to estimate volatility on the data
    windows : list of window lengths to use
    percentiles : percentiles ot calculate the cone values for given as a
                  number between 0 and 1.
    """
    percentiles = np.clip(percentiles,0,1)*100
    T = len(data)
    def adjustment(subseries_length):
        h = subseries_length
        n = T-h+1
        return np.sqrt(1/(1-h/n+(h*h-1)/(3*n*n)))
    
    cones = []
    windows_used = []
    current = []
    for h in windows:
        if (h > 0) and (h < T):
            adjFactor = adjustment(h)
            series = data.rolling(h).apply(estimator).dropna()
            cones.append(adjFactor*np.percentile(series,percentiles))
            current.append(series.iloc[-1][0])
            windows_used.append(h)

    res=pd.DataFrame(cones)
    res.index = windows_used
    res.columns = percentiles
    res['current'] = current
    return res