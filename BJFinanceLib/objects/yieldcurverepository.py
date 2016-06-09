# -*- coding: utf-8 -*-
        
class YieldCurveRepository():
    """ Class that centrally stores yield curves"""
           
    def __init__(self):
        self._curves = dict()
        
    def register(self,currency,descriptor,curve):
        """ registers a curve for a given currency and frequency with the
            repository.
            
            Arguments
            currency - currency for which the curve is
            descriptor - description for the curve
            curve - interest rate curve to be registered
        """
        if not currency in self._curves.keys():
            self._curves[currency] = dict()
        self._curves[currency][descriptor] = curve
        
    def currencies(self):
        """ Provides a list of all currencies available """
        return list(self._curves.keys())    
        
    def curves_for_currency(self,currency):
        """ Provides a list of all curves available for a given currency """
        if currency in self._curves.keys():
            return list(self._curves[currency].keys())
        else:
            return []
            
    def retrieve_curve(self,currency,descriptor):
        """ Returns a curve for given currency and frequency """
        try:
             return self._curves[currency][descriptor]
        except:
            return None