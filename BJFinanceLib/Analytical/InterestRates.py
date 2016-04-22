# -*- coding: utf-8 -*-
"""
Formulae for conversions on interest rates, discount factors, etc.

Author: Bram Jochems
Date: 20Apr16
"""

import numpy as np
from numpy import inf

def YieldToDiscountFactor(rate=None,ttm=None,comp=inf):
    """
    Converts a yield to a discount factor
      rate: the interest rate to convert
      ttm: the time to maturity to get the discount factor
      comp: the compounding frequency per year. Optional, default=inf, i.e.
            continuous compounding. Must be > 0.
    """
    ttm = np.maximum(ttm,0)
    if comp != inf:
        if comp > 0:
            return (1+rate/comp)**(-comp*ttm)
        else:
            raise ValueError("Compounding frequency must be > 0")
    else:
        return np.exp(-rate*ttm)
        
def DiscountFactorToYield(discountFactor=None,ttm=None,comp=inf):
    """
    Converts a discount factor to a yield
      discountfactor: the discount factor to convert. Must be > 0
      ttm: the time to maturity to get the discount factor
      comp: the compounding frequency per year. Optional, default=inf, i.e.
            continuous compounding. Must be > 0
    """
    if discountFactor <= 0:
        raise ValueError("Negative discountfactor not allowed")
    elif ttm <= 0:
        raise ValueError("Cannot compute yield for non-positive ttm")
    else:
        if comp != inf:
            return (discountFactor**(-1/(comp*ttm))-1)*comp
        else:
            return -np.log(discountFactor)/ttm

def ForwardRate(rate1=None,ttm1=None,rate2=None,ttm2=None,comp=inf):
    """
    Calculates the forward rate between two given rates and corresponding
    maturities for a given compounding frequency:
        rate1: the rate corresponding to ttm1
        ttm1: first maturity,  must be >= 0
        rate2: the rate corresponding to ttm2
        ttm2: second maturity, must be >= 0
        comp: compounding frequency per annum. Optional, default inf
              meaning continuous. Must be > 0.
    """
    if (ttm1==ttm2):
        raise ValueError("Cannot compute instantaneous forward rate")
    else:
        if (ttm1<=ttm2):
            ttm_min, rate_min = ttm1, rate1
            ttm_max, rate_max = ttm2, rate2
        else:
            ttm_min, rate_min = ttm2, rate2
            ttm_max, rate_max = ttm1, rate1
        if ttm_min < 0:
            raise ValueError("Cannot compute forward rate with negative ttm")
        elif ttm_min == 0:
            return rate2
        else:
            df1 = YieldToDiscountFactor(rate_min,ttm_min,comp)
            df2 = YieldToDiscountFactor(rate_max,ttm_max,comp)
            return DiscountFactorToYield(df2/df1,ttm_max-ttm_min,comp)
    