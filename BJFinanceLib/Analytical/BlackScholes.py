# -*- coding: utf-8 -*-
"""
Formulae for option values and greeks for various types of options in a
black-scholes world.

Author: Bram Jochems
Date: 20Apr16
"""
import BJFinanceLib.Analytical.InterestRates as IR
import numpy as np
from scipy.stats import norm
 
__spotNameList = ['spot','s']
__fwdNameList = ['forward','fwd','f']
__strikeNameList = ['strike','x','k']
__volNameList = ['vol','v','volatility','sigma']
__ttmNameList = ['t','ttm','maturity','time_to_maturity','timetomaturity']
__ccNameList = ['costofcarry','cc','cost_of_carry','b']
__rateNameList = ['rate','r','ir','interestrate','interest_rate']
__dfNameList = ['df','discount_factor','discountfactor']
__divyieldNameList = ['divyield','q','div_yield','dividendyield','dividend_yield']
__flagNameList = ['flag','cp','cp_flag']   

def __getElOrZero(lst,dic):
    val = [dic[key] for key in lst if key in dic.keys()]
    val.append(0)
    return val[0]

def __getRateOrZero(argDict):
    #potentially superfluous to lower again, but no need to optimize for now    
    argdictLower = dict([(k.lower(),v) for (k,v) in argDict.items()])
    if (np.any([el in argDict.keys() for el in __rateNameList])):
        rate = __getElOrZero(__rateNameList,argdictLower)
    elif (np.any([el in argDict.keys() for el in __dfNameList])) and \
         (np.any([el in argDict.keys() for el in __ttmNameList])):
        df = __getElOrZero(__dfNameList,argdictLower)
        t = __getElOrZero(__ttmNameList,argdictLower)
        if (df > 0) and (t>0):
            rate = IR.DiscountFactorToYield(df,t)
        else:
            raise ValueError("Cannot determine interest rate from given parameters")
    else:
        rate = 0
    return rate

def _d12(fwd,strike,vol,ttm):
    vsqrt = vol * np.sqrt(ttm)
    d1 = (np.log(fwd/strike) + 0.5*vol*vol*ttm)/vsqrt
    d2 = d1 - vsqrt
    return (d1,d2)

#Check if at least one of the required fields is available for eac
#required field
def __check_required(lst,dic, missing_param_des):
    if not (np.any([el in dic.keys() for el in lst])):
        raise KeyError("Missing a value for " + missing_param_des)

  

def forward(spot,ttm,**kwargs):
    """
    Calculates the forward given spot, cost of carry and time-to-maturity. This
    is actually a model-indepenent  relationship, but fits well within this
    module.
      spot: spot to sue
      ttm: the time to maturity to get the discount factor
      kwargs: dictionary that takes various arguments to specify how the
              forward should be constructed. It accepts either values that
              specify a cost of carry or a combination of optional interest
              rate and dividend yield parameters. If the latter aren't provided
              they are assumed to be zero. Overparametrizing the arguments will
              not give an error, but should be avoided. Accepted keys for various
              parameters (all not case sensitive)
                 costofcarry - 'cc','costofcarry','b', 'cost_of_carry'
                 interest rate - 'rate', 'r', 'interest_rate', 'ir'
                 dividend yield - 'divyield','div_yield','dividend_yield','q'
    """
    kwlower = dict([ (k.lower(),v) for (k,v) in kwargs.items() ])    
    if np.any([el in kwlower.keys() for el in __ccNameList]):
        cc = __getElOrZero(ccNameList,kwlower)
        #lookup in dictionary won't fail due to if clause
    else:
        rate = __getRateOrZero(kwlower)
        divyield = __getElOrZero(__divyieldNameList,kwlower)
        cc = rate-divyield
    return spot*np.exp(cc*ttm)

def validateBSArgDict(**argDict):
    """ Validatesa dictioanry of parameters for black-scholes functions.
        Returns a tuple consisting of (fwd,strike,vol,ttm,df,flag). Some
        missing paramters will be inferred """
        
    argdictLower = dict([(k.lower(),v) for (k,v) in argDict.items()])
 
    __check_required(__spotNameList+__fwdNameList,argdictLower,'spot or forward')
    __check_required(__volNameList,argdictLower,'volatility')
    __check_required(__ttmNameList,argdictLower,'time to maturity')
    
    vol = __getElOrZero(__volNameList,argdictLower)
    ttm = __getElOrZero(__ttmNameList,argdictLower)

    if (vol <= 0) or (ttm <= 0):
        return 0

    # determine forward
    if (np.any([el in argdictLower.keys() for el in __fwdNameList])):
        fwd = __getElOrZero(__fwdNameList,argdictLower)
    else:
        spot = __getElOrZero(__spotNameList,argdictLower)
        fwd = forward(spot,ttm,**argDict)

    #determine strike
    if (np.any([el in argdictLower.keys for el in __strikeNameList])):
        strike = __getElOrZero(__strikeNameList,argdictLower)
    else:
        strike = fwd #If no strike supplied, assume ATM
        
    # flag for call or put
    if (np.any([el in argdictLower.keys for el in __flagNameList])):
        flag = __getElOrZero(__flagNameList,argdictLower).lower()[0]
        if flag == 'c':
            eta = 1.
        elif flag == 'p':
            eta = -1.
        else:
            raise ValueError("Incorrect option flag type")
    else:
        if fwd < strike:
            eta = 1.
        else:
            eta = -1.

    # discount factor
    if (np.any([el in argdictLower.keys for el in __dfNameList])):
        df = __getElOrZero(__dfNameList,argdictLower)
    else:
        rate = __getRateOrZero(argdictLower)
        df = IR.YieldToDiscountFactor(rate,ttm)

    return (fwd,strike,vol,ttm,df,eta)

    
def value(flag=None,**kwargs):
    """
    Calculates the value of an european vanilla optoin in the black-scholes
    setting.
        kwargs: the other parameter that need to be specified to calcualte the
                option value. At the bare minimum, a value for either spot or
                forward needs to be provided, as well as a time to maturity,
                strike and volatility. Additionally, interest rate,
                dividend yield and costOfCarry may be provided, and a call or
                put flag
    """
    (fwd,strike,vol,ttm,df,flag) = validateBSArgDict(kwargs)
                
                
                
