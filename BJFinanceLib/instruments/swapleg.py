# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import datetime
from numbers import Number
from numpy import inf
from BJFinanceLib.objects.rates import YieldCurve, ForwardRate
from BJFinanceLib.objects.cashflowschedule import CashflowSchedule
from BJFinanceLib.objects.swapschedule import SwapSchedule

class SwapLeg(ABC):

    def __init__(self,notional,schedule,currency=None,dayCounter=None):
        """ Initializes IRSLeg
        
        Arguments
            - notional : notional for the leg
            - schedule : SwapSchedule of periods for swap leg
            - currency: currency for the leg
            - daycounter : daycounter for the leg
        """
        self.schedule = None
        """  Schedule of dates for the swap leg """
        if isinstance(schedule,SwapSchedule):
            self.schedule = schedule
        else:
            try:
                self.schedule = SwapSchedule(schedule)
            except:
                raise Exception('Cannot process swap schedule for swap leg')
                
        self.notional = notional
        """ Notional for the swap leg """
        
        self.currency = currency
        """ currency for swap leg"""
        
        def defaultDayCounter(first,second):
            """ default daycounter function if none is provided """
            if isinstance(first,datetime.date):
                return (second-first).days/365
            elif isinstance(first,Number):
                return (second-first)
            else:
                raise Exception('Unsupported type for default daycounter')
        
        self.daycounter = dayCounter or defaultDayCounter
        """ day counter for the curve. If non is specified, a simple Act/365
            is used as default """
    
    @abstractmethod
    def get_rate(self,startDate,endDate,referenceDate):
        """ Method that returns the applicable swap rate for a swap leg for a
            period that starts at startDate and ends at endDate."""
        pass

    def cashflow_schedule(self,referenceDate):
        """ Returns the unsettled cash flow schedule for a reference date """
        dates = self.schedule.unsettled_periods(referenceDate)
        datedCashFlows = [(settle, self.daycounter(start,end) * 
                                   self.notional * 
                                   self.get_rate(start,end,referenceDate))
                          for (start,end,settle) in dates]
        dates, cashflows = zip(*datedCashFlows)
        return CashflowSchedule(dates,cashflows,self.currency)
    
    def present_value(self,referenceDate,discountFunction):
        """ Calculates the present value of a swap leg on a given reference
            date and discount function.
            
            Arguments
            - referenceDate : the date for which to do the valuation
            - discountFunction: callable object that returns for a given swap
                                date the discountfactor.
        """
        cashflowSchedule = self.cashflow_schedule(referenceDate)
        pvs = [self.discountFunction(date)*amount for
                (date,amount) in cashflowSchedule.dates_with_amounts.items()]
        return sum(pvs)


class SwapLegFixed(SwapLeg):
    """ Represents a leg for a swap that has a fixed rate"""
    def __init__(self,notional,schedule,fixed_rate,currency=None,dayCounter=None):
        """
        Arguments
           - notional : notional for the swap leg
           - schedule : swapschedule for the swap leg
           - currency : currency for the swap leg
           - fixed_rate : fixed rate for the swap leg
           - daycounter : day count function for the swap leg. Optional, if
                          not provided the base class provides a default.
        """
        super(SwapLegFixed,self).__init__(notional,schedule,currency,dayCounter)
        self.fixed_rate = fixed_rate
    
    def get_rate(self,startDate,endDate,referenceDate):
        """ Returns the annualized fixed rate from startdate to enddate"""
        return self.fixed_rate
    
    
class SwapLegFloating(SwapLeg):
    """ Represent a leg for a swap that has a floating amount with a spread
        added to it """
    def __init__(self,notional,schedule,yieldFunction,spread=0,currency=None,dayCounter=None):
        """
        Arguments
           - notional : notional for the swap leg
           - schedule : swapschedule for the swap leg
           - currency : currency for the swap leg
           - yieldFunction : function that provides a yield when provided with
                             a date or time to maturity.
           - spread : additional spread to apply to each period's rate
           - daycounter : day count function for the swap leg. Optional, if
                          not provided the base class provides a default.
        """
        super(SwapLegFloating,self).__init__(notional,schedule,currency,dayCounter)
    
        self.floating_leg_yield_function = yieldFunction    
        """ function that calculates yield for each swap period"""
        
        self.spread = spread    
        """ annualized spread to be added to each swap period"""
    
    def get_rate(self,startDate,endDate,referenceDate):
        """ Retrieves the rate for a floating leg for a given period and
            reference date"""
        yc = self.floating_leg_yield_function
        if isinstance(yc,YieldCurve):
            return ForwardRate(yc.interest_rate(referenceDate,startDate),
                               self.daycounter(startDate),
                               yc.interest_rate(referenceDate,endDate),
                               self.daycounter(endDate),
                               yc.compounding_frequency) + self.spread
        elif hasattr(yc,'__call__'):
             return ForwardRate(yc(startDate),
                               self.daycounter(referenceDate,startDate),
                               yc(endDate),
                               self.daycounter(referenceDate,endDate),
                               inf) + self.spread
        else:
            raise Exception('Unsupported floating leg yield function in SwapLegFloating')
