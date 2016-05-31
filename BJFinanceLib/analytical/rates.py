# -*- coding: utf-8 -*-
"""
Objects to represent yield curves

@author: Bram Jochems
"""
from abc import ABC, abstractmethod
import datetime
from numbers import Number
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

class YieldCurve(ABC):
    """
    Base class for yield curves. Yield curve objects provide access to
    functionality to retrieve discountfactors and interest rates. Classes that
    derive from this class must implement the abstract method
    _internal_df_calc from which other calculations are derived (unless)
    overriden
    """
    
    def __init__(self,
                 referenceDate=None,
                 dayCounter=None,
                 compoundingFrequency=None):
        """ Constructor for base class for yield curves.
        
        Arguments:
        referenceDate: reference date for which the curve is. Optional, if None
                       is specified, callling this class for an interest rate
                       or discount factor for another date will return an error
                       If this is specified, these function do work.
        dayCounter: function that returns the difference between two dates as
                    a yearfraction. Optional, if none is specified, a simple
                    Act/365 is used.
        compoundingFrequency: the compoundingFrequency at which the curve is
                              specified. Optional. If none, infinite is assumed
                              meaning continuous compounding.
        """

        self._accept_dates = referenceDate and isinstance(referenceDate,datetime.date)        
        """ True if methods can accept dates as input """
        
        self.reference_date = referenceDate
        """ The reference date for the curve """
        
        self.daycounter = dayCounter or (lambda first,second: (second-first).days/365)
        """ day counter for the curve. If non is specified, a simple Act/365
            is used as default """
            
        self.compounding_frequency = compoundingFrequency or inf
        """ compounding frequency for the curve """
    
    @abstractmethod
    def _internal_df_calc(self,ttm):
        """ Method that does the actual work of calculating a discount factor """
        pass
    
    def _ttm(self,date):
        """ Calculates the ttm associated to a date for the yieldcurve.
        
        Argument:
        date : either a date or a time to maturity. If a date is specified,
               the time to maturity is calculated internally, if on construction
               the curve's referenceDate parameter was specified. If it wasn't,
               providing a date will result in an error.
               
        Returns: the time to maturity associated to a date.
        """
        if isinstance(date,datetime.date):
            if self._accept_dates:
                return self.daycounter(self.reference_date,date)
            else:
                raise "Yieldcurve object doesn't accept dates as input"
        elif isinstance(date,Number):
            return date
        else:
            raise "Unrecognized date type"
    
    def discount_factor(self,date):
        """ retrieves a discount factor for a given date
        Argument:
        date : either a date or a time to maturity. If a date is specified,
               the time to maturity is calculated internally, if on construction
               the curve's referenceDate parameter was specified. If it wasn't,
               providing a date will result in an error.
               
        Returns: discountfactor associated with date
        """
        ttm = self._ttm(date)
        return self._internal_df_calc(ttm)
        
    def interest_rate(self,date):
        """ Retrieve the interest rate associated to a date """
        df = self.discount_factor(date)
        ttm = self._ttm(date)
        return DiscountFactorToYield(df,ttm,self.compounding_frequency)
        
    def forward_rate(self,date1,date2):
        """ retrieves the forward rate from date1 to date2
    
        Argument:
        date1 : The first "date". Either a date or a time to maturity. If a
                date is specified, the time to maturity is calculated
                internally, if on construction the curve's referenceDate
                parameter was specified. If it wasn't, providing a date will
                result in an error.
        date2 : The second "date". Either a date or a time to maturity. If a
                date is specified, the time to maturity is calculated
                internally, if on construction the curve's referenceDate
                parameter was specified. If it wasn't, providing a date will
                result in an error. Date 2 must be larger than date 1 
        Returns: forward rate from date1 to date 2.
        """        
        ttm1 = self._ttm(date1)
        ttm2 = self._ttm(date2)
        rate1 = self.interest_rate(ttm1)
        rate2 = self.interest_rate(ttm2)
        return ForwardRate(rate1,ttm1,rate2,ttm2,self.compounding_frequency)
    
    def __add__(self,other):
        """ Adds two yield curves to form a new one """
        return YieldCurveSum(self,other,
                             referenceDate=self.reference_date,
                             dayCounter= self.daycounter,
                             compoundingFrequency=self.compounding_frequency)
       
class YieldCurveOnDF(YieldCurve):
    """ Yield curve object that does interpolation on discountfactors """
    def __init__(self,
                 interpolator,
                 definingPoints=None,
                 referenceDate=None,
                 dayCounter=None,
                 compoundingFrequency=None):
        """
        Instantiates a yield curve object that interpolates on discount factor
        
        Arguments
        interpolator: function that does the interpolation on the discount-
                      factors. If definingPoints is also defined, then this
                      function takes as an input 3 argument, a list of old x
                      values, a list of old y values and a new x value. If
                      definingPoints is not defined, then this takes a single
                      new x value as input, effectively allowing a parametric
                      discount factor specification. If the interpolator is set
                      to None and definingPionts are provided, piecewise linear
                      interpolation is used (which migth give unexpected
                      results when extrapolating.)
        definingPoints: points on which interpolation happens. Should be of the
                        form of an interable of tuples, where the first element
                        is a date of a time to maturity and the second is a
                        discount factor. Optional, if not specified, the
                        interpolation function is just used as a parametric
                        specfication of the yield.
        referenceDate: reference date for which the curve is. Optional, if None
                       is specified, callling this class for an interest rate
                       or discount factor for another date will return an error
                       If this is specified, these function do work.
        dayCounter: function that returns the difference between two dates as
                    a yearfraction. Optional, if none is specified, a simple
                    Act/365 is used.
        compoundingFrequency: the compoundingFrequency at which the curve is
                              specified. Optional. If none, infinite is assumed
                              meaning continuous compounding. 
        """
        super(YieldCurveOnDF,self).__init__(referenceDate,dayCounter,compoundingFrequency)

        self.defining_points = definingPoints
        """ The points - if any - on which interpolation occures """

        # Create an interpolator as a bound method        
        if definingPoints == None:
            # No defining points, interpolator is parameteric
            self._interpolator = lambda x: interpolator(x)
        else:
            # Defining points. First make sure that they are unique by ttm,
            # then sort then, unzip them unzip them, then map the x_old values
            # to ttms and then create the bound method for the interpolator
            seen = set()
            points = [item for item in definingPoints if item[1] not in seen and not seen.add(item[1])]
            x_old,y_old = list(zip(*sorted(points)))
            x_old = [self._ttm(x) for x in x_old]
            
            if interpolator:                    
                self._interpolator = lambda x: interpolator(x_old,y_old,x)
            else:
                # default to piecewise linear interpolation on input points
                self._interpolator = lambda x: np.interp(x,x_old,y_old)
         
    def _internal_df_calc(self,ttm):
        """ Method that does the actual work of calculating a discount factor.
            Implementation of the abstract method in the YieldCurve base class
        """     
        return self._interpolator(ttm)
        
class YieldCurveOnIR(YieldCurve):
    """ Yield curve object that does interpolation on rates directly """
    def __init__(self,
                 interpolator,
                 definingPoints=None,
                 referenceDate=None,
                 dayCounter=None,
                 compoundingFrequency=None):
        """
        Instantiates a yield curve object that interpolates on rates
        
        Arguments
        interpolator: function that does the interpolation on the interest
                      rates. If definingPoints is also defined, then this
                      function takes as an input 3 arguments, a list of old x
                      values, a list of old y values and a new x value. If
                      definingPoints is not defined, then this takes a single
                      new x value as input, effectively allowing a parametric
                      discount factor specification. If the interpolator is set
                      to None and definingPionts are provided, piecewise linear
                      interpolation is used (which migth give unexpected
                      results when extrapolating.)
        definingPoints: points on which interpolation happens. Should be of the
                        form of an interable of tuples, where the first element
                        is a date of a time to maturity and the second is a
                        interest rate. Optional, if not specified, the
                        interpolation function is just used as a parametric
                        specfication of the yield.
        referenceDate: reference date for which the curve is. Optional, if None
                       is specified, callling this class for an interest rate
                       or discount factor for another date will return an error
                       If this is specified, these function do work.
        dayCounter: function that returns the difference between two dates as
                    a yearfraction. Optional, if none is specified, a simple
                    Act/365 is used.
        compoundingFrequency: the compoundingFrequency at which the curve is
                              specified. Optional. If none, infinite is assumed
                              meaning continuous compounding. 
        """
        super(YieldCurveOnIR,self).__init__(referenceDate,dayCounter,compoundingFrequency)
        
        self.defining_points = definingPoints
        """ The points - if any - on which interpolation occures """

        # Create an interpolator as a bound method        
        if definingPoints == None:
            # No defining points, interpolator is parameteric
            self._interpolator = lambda x: interpolator(x)
        else:
            # Defining points. First make sure that they are unique by ttm,
            # then sort then, unzip them unzip them, then map the x_old values
            # to ttms and then create the bound method for the interpolator
            seen = set()
            points = [item for item in definingPoints if item[1] not in seen and not seen.add(item[1])]
            x_old,y_old = list(zip(*sorted(points)))
            x_old = [self._ttm(x) for x in x_old]
            
            if interpolator:                 
                self._interpolator = lambda x: interpolator(x_old,y_old,x)
            else:
                # default to piecewise linear interpolation on input points
                self._interpolator = lambda x: np.interp(x,x_old,y_old)        
        
    def _internal_df_calc(self,ttm):
        """ Method that does the actual work of calculating a discount factor.
            Implementation of the abstract method in the YieldCurve base class
        """
        ttm = self._ttm(ttm)
        rate = self.interest_rate(ttm)
        return YieldToDiscountFactor(rate,ttm,self.compounding_frequency)

    def interest_rate(self,date):
        """ Since this class interpolates on rates directly, override of the
            base class method for determining interest rates linked to df's """
        ttm = self._ttm(date)
        return self._interpolator(ttm)
        
class YieldCurveTriangle(YieldCurve):
    """
    Yield curve that is flat with the exception of a 'triangle' where it
    assumes a value. Useful for calculating sensitivities.
    """
    def __init__(self, xMid, yMid = 0.001, xLeft=None, yLeft=0, xRight=None,
                 yRight=0, referenceDate=None, dayCounter=None,
                 compoundingFrequency=None):    
        """
        Constructor for YieldCurveTriangle class. This class is a yieldcurve
        with the following interest rate profile:
        t <= xLeft, a value yLeft is assumed. For xLeft < t < xMid, a linear
        interpolation between yLeft and yMid is done. For xMid <= t < xRight,
        linear interpolation between yMid and yRight is done. for t >= xRight,
        yRight is assumed.
        
        Arguments:
        xLeft : the left-cutoff point for defining the bump. If left out,
                it is assumed that yLeft = yMid.
        yLeft: the value that the curve takes on to the left of xLeft.
               Optional, default=0 when xLeft is specified and if xLeft
                is None, then this defaults to xMid.
        xMid: the value where the triangle peak is located
        yMid: the value of the peak of the triangle. Optional, default =
              0.001 (10 basis points)
        xRight: the right cutoff point for defining the bump. If left out,
                it is assumed that yRight = yMid
        yRight: the value that the cuves takes on to the right of xRight.
                Optional, default=0 when xRight is specified and if xRight
                is None, then this defaults to xMid.
        referenceDate: reference date for which the curve is. Optional, if None
                       is specified, callling this class for an interest rate
                       or discount factor for another date will return an error
                       If this is specified, these function do work.
        dayCounter: function that returns the difference between two dates as
                    a yearfraction. Optional, if none is specified, a simple
                    Act/365 is used.
        compoundingFrequency: the compoundingFrequency at which the curve is
                              specified. Optional. If none, infinite is assumed
                              meaning continuous compounding.
                          
        Note that not specifying xLeft and xRight both leads to a parallel
        shift. In such a case any value can be used for xMid.
        """
        super(YieldCurveTriangle,self).__init__(referenceDate,dayCounter,compoundingFrequency)
        
        pt1 = [(xLeft,yLeft)] if xLeft else []
        pt2 = [(xMid,yMid)]
        pt3 = [(xRight,yRight)] if xRight else []
        self.defining_points = sorted(pt1+pt2+pt3)
        """ The points that defing the triangle bump """
        
        x_old,y_old = zip(*self.defining_points)   
        self._interpolator = lambda x: np.interp(x,x_old,y_old)        
        """ interpolator object that is used internally """
        
    def _internal_df_calc(self,ttm):
        """ Method that does the actual work of calculating a discount factor.
            Implementation of the abstract method in the YieldCurve base class
        """
        ttm = self._ttm(ttm)
        rate = self.interest_rate(ttm)
        return YieldToDiscountFactor(rate,ttm,self.compounding_frequency)

    def interest_rate(self,date):
        """ Since this class interpolates on rates directly, override of the
            base class method for determining interest rates linked to df's """
        ttm = self._ttm(date)
        return self._interpolator(ttm)
        
class YieldCurveSum(YieldCurve):
    """
    Yield curve object that is the sum of two other yieldcurves
    """
    
    def __init__(self, curve1, curve2, referenceDate=None, dayCounter=None,
                 compoundingFrequency=None): 
        """
        Instantiates a yield that is the sum of two yield curves. Note that
        this object calls the discount factor methodology from each curve
        seperately, but if there are differnece between the daycounters,
        reference dates or compounding frequencies, results might be unexpected
        
        Arguments
        curve1: first of the two curves to be added.
        curve2: second curve added.
        referenceDate: reference date for which the curve is. Optional, if None
                       is specified, callling this class for an interest rate
                       or discount factor for another date will return an error
                       If this is specified, these function do work.
        dayCounter: function that returns the difference between two dates as
                    a yearfraction. Optional, if none is specified, a simple
                    Act/365 is used.
        compoundingFrequency: the compoundingFrequency at which the curve is
                              specified. Optional. If none, infinite is assumed
                              meaning continuous compounding. 
        """
        super(YieldCurveSum,self).__init__(curve1.reference_date,
                                           curve1.daycounter,
                                           curve1.compounding_frequency)
                                           
        self._basecurve=curve1
        self._addcurve=curve2

    def _internal_df_calc(self,ttm):
        """ Method that does the actual work of calculating a discount factor """
        return self._basecurve._internal_df_calc(ttm) * self._addcurve._internal_df_calc(ttm)
        
class YieldCurveRepository():
    """ Class that centrally stores yield curves"""
    
    _allowed_frequences_str =  ['ois','OIS','1m','3m','6m','12m','1M','3M','6M','12M']  
    _allowed_frequencies_int =  [0,1,3,6,12]
        
    def __init__(self):
        self._curves = dict()
        
    @staticmethod
    def _freq_to_string(freq):
        if freq in YieldCurveRepository._allowed_frequencies_int:
            if freq == 0:
                return 'OIS'
            else:
                return str(freq) + 'M'
        elif freq in YieldCurveRepository._allowed_frequencies_str:
            return freq.upper()
        else:
            raise Exception("Invalid curve frequency")
        
    def register(self,currency,frequency,curve):
        """ registers a curve for a given currency and frequency with the
            repository.
            
            Arguments
            currency - currency for which the curve is
            frequency - frequency/description for the curve, allowed values are
                        in the _allowed_frequencies_str and
                        _allowed_frequencies_int class attributes
            curve - interest rate curve to be registered
        """
        if not currency in self._curves.keys():
            self._curves[currency] = dict()
        freq = self._freq_to_string(frequency)
        self._curves[currency][freq] = curve
        
    def currencies(self):
        """ Provides a list of all currencies available """
        return list(self._curves.keys())    
        
    def curves_for_currency(self,currency):
        """ Provides a list of all curves available for a given currency """
        if currency in self._curves.keys():
            return list(self._curves[currency].keys())
        else:
            return []
            
    def retrieve_curve(self,currency,frequency):
        """ Returns a curve for given currency and frequency """
        try:
            freq = self._freq_to_string(frequency)
            return self._curves[currency][freq]
        except:
            return None