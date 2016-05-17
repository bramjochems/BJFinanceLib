from unittest import TestCase

import math
import BJFinanceLib.analytical.blackscholes as BS

class ValueTests(TestCase):
    
    def putCallParityHolds(self):
        """
        Test that acertains that for an option that is ATMF, call and put values
        are the same (for a given set of paramters)
        """
        spot, costOfCarry, ttm, df, vol = 100, 0.03, 1, 1, 0.20
        strike = spot*math.exp(costOfCarry*ttm)
        callValue = BS.GeneralizedBlackScholes('c',spot=100,strike=strike,ttm=ttm,
                                               discountFactor=df, volatility=vol,
                                               costOfCarry = costOfCarry)
        putValue = BS.GeneralizedBlackScholes('p',spot=100,strike=strike,ttm=ttm,
                                               discountFactor=df, volatility=vol,
                                               costOfCarry = costOfCarry)
        self.assertTrue(math.abs(putValue-callValue) < 1e-8)    
    
    def solvingForMissingParamterRecoversIt(self):
        """
        Test that omitting one value from the pricer and retrieving it via
        seaching on the option's price returns it (within reasonable numerical
        precision)
        """
        args = { 'spot' : 100, 'costOfCarry' : 0.02, 'ttm' : 3,
                 'discountFactor': 0.94, 'volatility' : 0.20, 'strike' : 120 }       
        callValue = BS.GeneralizedBlackScholes('c',**args)
        res = []                               
        for arg in args.keys():
            tmpDict = args.copy()
            argReal = args[arg]
            del args[arg]
            argFound = BS.GeneralizedBlackScholes('c',price=callValue,**tmpDict)
            res.append(math.abs(argReal-argFound) < 1e-8)
        self.assertTrue(not (False in res))

    def OptionValuesAreAsExpected(self):
        pass
    
    def OptionDeltasAreAsExpected(self):
        pass
    
    def OptionGammasAreAsExpected(self):
        pass
    
    def OptionVegasAreAsExpected(self):
        pass
    
    def NonPositiveTtmGivesZero(self):
        pass
    
    def NegativeSpotGivesError(self):
        pass
    
    def NegativeVolatilityGivesError(self):
        pass
    
    def NegativeDiscountFactorGivesError(self):
        pass