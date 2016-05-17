# -*- coding: utf-8 -*-
import math
import scipy.stats as ss
import scipy.optimize as opt

def _flagToInt(flag):
    """
    Helper function that turns a flag into a +/-1 float.
    """
    flag =  flag.strip().lower()[0]
    if flag == 'c':
        return 1.0
    elif flag == 'p':
        return -1.0
    else:
        raise Exception('Invalid flag')
 
def _d1d2ByFwd(forward,strike,vol,ttm):
    """
    Helper function to calculate d1 and d2
    """
    ssqrt = vol * math.sqrt(ttm)
    d1 = (math.log(forward/strike))/ssqrt + 0.5*ssqrt
    d2 = d1 - ssqrt
    return (d1,d2)
 
def forward(spot=None, costOfCarry=None,ttm=None):
    """
    Forward from spot
    """
    return spot*math.exp(costOfCarry*ttm)
 
def GeneralizedBlackScholes(flag=None,spot=None,strike=None,volatility=None,
                            ttm = None, discountFactor=None, costOfCarry=None,
                            price=None,output=None):
    """
    Generalized black scholes formula. Leave one parameter blank and it solves
    for this parameter.
    
    Parameters:
        - flag: string indicating a 'c(all)' or 'p(ut)'
        - spot : spot price of the underlying asset
        - strike : strike  price for the option
        - volatility : implied volaitlity to use
        - ttm : time to maturity for the option expiration     
        - discountFactor : discountFactor associated with the ttm
        - costOfCarry : costOfCarry as a continuous yield
        - price: the price to use when solving for another paramter
        - output: a list of outputs that can be request. If output=None, then
                  it is substituted by a list containing t he field ['value'].
                  Other arguments that can be entered into, or added to this
                  list are delta, gamma, vega, gammaP, theta, rho, carryRho,
                  d1, d2.
                  A few remarks on those:
                  - vega, rho, carryRho for a 1% (absolute) change in the
                    the related paramter
                  - theta is per 1/365th change in time to maturity
                      
    Returns:
        - A single option value or greek if all parameters are specified and
          only a single output is requested
        - A list of option value and greeks if all paramters are specified and
          multiple outputs were requested
        - A single parameter if an option value was provided and an input
          paramter was missing
    """
    # NOTE: I've gone back and forth a few times between specifying spot and
    #       costOfCarry as input parameters versus just the forward a few times.
    #       The reasons to settle down on spot + costOfCarry (which is some
    #       sense is less canonicol to me than the forward, is that your for
    #       example your delta then also becomes by the forward. First of all,
    #       this isn't alwyas intuitive/what people would expect. Secondly, not
    #       all cases are disambiguous. An option that is deep in the money, might
    #       still have a delta equal to the discount factor or a delta equal to
    #       1. E.g. an option on a future, will typically have a delta equal to
    #       the discount factor, while an option on a share, will have a delta
    #       (by forward) of one.)
    flag = _flagToInt(flag)
    parameters = locals().copy()

    def _black(flag, spot, strike, ttm, discountFactor, volatility,
               costOfCarry, price, output, calibration=False, **kwargs):
        """
        helper function that does the actual black-scholes calculations.
        Depending on inputs, this might return just an option value or a
        dicitonary of values
        """
        ebt =math.exp(costOfCarry*ttm)
        forward = spot * ebt
        d1, d2 = _d1d2ByFwd(forward,strike,volatility,ttm)
        Nd1f,Nd2f = ss.norm.cdf(flag*d1), ss.norm.cdf(flag*d2)
        price = discountFactor*(flag*forward*Nd1f - flag * strike * Nd2f)
        
        if not calibration and output:
            if not ttm <=0.0:
                sqrtt = math.sqrt(ttm)
                res = {}
                nd1 = ss.norm.pdf(d1)
                if "price" in output: res["price"] = price
                if "delta" in output: res["delta"] = flag * ebt * discountFactor * Nd1f
                if "gamma" in output: res["gamma"] = discountFactor * ebt * nd1/(spot*volatility*sqrtt)
                if "gammaP" in output: res["gammaP"] = discountFactor * ebt * nd1/(100*volatility*sqrtt) #gamma * spot/100   
                if "vega" in output: res["vega"] = discountFactor * forward*nd1*sqrtt/100.0
                if "theta" in output:
                    r = math.log(discountFactor)/-ttm
                    term1 = discountFactor*forward * nd1 * volatility/(2.0*sqrtt)
                    term2 = flag*(costOfCarry-r)*discountFactor*forward*Nd1f
                    term3 = flag*r*strike*discountFactor*Nd2f
                    res["theta"] = -(term1+term2+term3)/365    
                if "rho" in output: res["rho"] = 0.01*flag*ttm*strike*discountFactor*Nd2f
                if "carryRho" in output: res["carryRho"] = 0.01*flag*ttm*forward*discountFactor*Nd1f
                if "d1" in output: res["d1"] = d1
                if "d2" in output: res["d2"] = d2            
                
                if len(res.keys()) == 1:
                    return [value for _,value in res.items()][0]    
                else:
                    return res
            else: return 0.
        else:
            return price

    def _calibrate(value, field, parameters):
        """
        Helper function that addes a field to parameters with a given value
        and then calculates the absolute difference between the resulting
        black scholes value and a prior specified black-scholes price
        """
        parameters.update({field: value[0]})
        return abs(_black(calibration=True, **parameters) - price, )

    # Check for missing parameters
    missing = [param for param in ["spot", "strike", "volatility", "ttm",
                                   "discountFactor", "price","costOfCarry"]
                               if parameters[param] is None]
    
    if len(missing) > 1:
        raise Exception("Too many missing variables from: %s " % missing)
    if len(missing) == 0:
        raise Exception("All variables assigned - nothing to solve")
    if missing[0] != "price":
        # if we are missing any parameter different from price we need to calibrate
        result = opt.fsolve(_calibrate, 0.1, args=(missing[0], parameters))
        if not output:
            return result[0]  # If output=None or [], simply return the calibrated parameter
    return _black(**parameters)       