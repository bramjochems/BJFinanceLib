# -*- coding: utf-8 -*-
"""
Shift lognormal (and anti-lognormal with the right paramer choices) valuation
"""
from BJFinanceLib.analytical.blackscholes import GeneralizedBlackScholes
import scipy.optimize as opt

def ShiftedLogNormal(flag=None,forward=None,shift=None,strike=None,
                     volatility=None, ttm = None, discountFactor=None,
                     price=None, output=None):
    """
    Displaced lognormal (and anti-lognormal) pricing. Assumes dynamics of the
    form
        d forward = volatility (forward + shift) dW
    
    Note the sign of the shift, different papers apply differnet conventions.
    In contrast to the GeneralizedBlackScholes function implemented, this
    works on the forward, not spot, since the only way the shifted lognormal
    remains tractable is if there's no drift (I think, actually not looked into
    that).
    
    This function internall calls jsut the GeneralizedBlackScholes function
    with shifted arguments. This works, but with a few caveats:
        - Not all shifts lead to valid sets of paramters. For invalid
          paramters, a numerical error might(!) follow

    """
    parameters = locals().copy()
    missing = [param for param in ["forward", "strike", "volatility", "ttm",
                                   "discountFactor", "price","shift"]
                               if parameters[param] is None]
    
    if len(missing) > 1:
        raise Exception("Too many missing variables from: %s " % missing)
    if len(missing) == 0:
        raise Exception("All variables assigned - nothing to solve")

    def _calibrate(shift):
        target = parameters['price']
        bs = GeneralizedBlackScholes(flag=parameters['flag'],
                                     spot=parameters['forward']+shift,
                                     strike=parameters['strike']+shift,
                                     ttm=parameters['ttm'],
                                     volatility=parameters['volatility'],
                                     costOfCarry=0,
                                     discountFactor=parameters['discountFactor'],
                                     price=None,
                                     output=None)
        return abs(target-bs)
        
        
    # If shift is missing, we need to solve for it first and then return what's
    # needed. If shift is not missing, then we can use the
    # GeneralizedBlackScholes, but if we're solving for shift or forward, we
    # need to take care what we are returning
    if missing[0] == "shift":
        # calibrate on shift and then see if we return 
        res = opt.fsolve(_calibrate,0)
        if output==None:
            return res[0]
        
    elif missing[0] == "forward":
        result =  GeneralizedBlackScholes(flag=flag, spot=None, strike=strike+shift,
                                       ttm=ttm, volatility=volatility,
                                       costOfCarry=0,
                                       discountFactor=discountFactor,
                                       price=price,output=output)
        return result[0]-shift if output==None else result
        
    elif missing[0] == "strike":
        result =  GeneralizedBlackScholes(flag=flag, spot=forward+shift,
                                          strike=None, ttm=ttm,
                                          volatility=volatility,
                                          costOfCarry=0,
                                          discountFactor=discountFactor,
                                          price=price,output=output)
        return result[0]-shift if output==None else result
        
    # Else case + fallthrough case for output != None if shift is missing
    return GeneralizedBlackScholes(flag=flag, spot=forward+shift,
                                   strike=strike+shift, ttm=ttm,
                                   volatility=volatility,
                                   costOfCarry=0,
                                   discountFactor=discountFactor,
                                   price=price,output=output)