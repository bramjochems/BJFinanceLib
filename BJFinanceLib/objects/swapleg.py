# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from BJFinanceLib.objects.cashflowschedule import CashflowSchedule
from BJFinanceLib.objects.swapschedule import SwapSchedule

class SwapLeg(ABC):

    def __init__(self,notional,schedule,curreny,dayCounter=None):
        """ Initializes IRSLeg
        
        Arguments
            - notional : notional for the leg
            - schedule : SwapSchedule of periods for swap leg
        """
        self.schedule = schedule
        """  Schedule of dates for the swap leg """
        
        self.notional = notional
        """ Notional for the swap leg """
        
        self.currency = currency
        """ currency for swap leg"""
        
        self.daycounter = dayCounter or (lambda first,second: (second-first).days/365)
        """ day counter for the curve. If non is specified, a simple Act/365
            is used as default """
    
    @abstractmethod
    def get_rate(self,startDate,endDate):
        """ Method that returns the applicable swap rate for a swap leg for a
            period that starts at startDate and ends at endDate."""
        pass

    def cashflow_schedule(self,referenceDate):
        """ Returns the unsettled cash flow schedule for a reference date """
        dates = self.schedule.unsettled_periods(referenceDate)
        datedCashFlows = [(settle, self.daycounter(start,end) * 
                                   self.notional * 
                                   self.get_rate(start,end))
                          for (start,end,settle) in dates]
        dates, cashflows = zip(*datedCashFlows)
        return SwapSchedule(dates,cashflows,self.currency)
    
    def present_value(self,referenceDate, discountFunction):
        """ Calculates the present value of a swap leg on a given reference
            date and discount function.
            
            Arguments
            - referenceDate : the date for which to do the valuation
            - discountFunction: callable object that returns for a given swap
                                date the discountfactor.
        """
        cashflowSchedule = self.cashflow_schedule(referenceDate)
        pvs = [discountFunction(date)*amount for
                (date,amount) in cashflowSchedule.dates_with_amounts.items()]
        return sum(pvs)


class SwapLegFixed(SwapLeg):
    """ Represents a leg for a swap that has a fixed rate"""
    def __init__(self,notional,schedule,currency,fixed_rate,dayCounter=None):
        super(SwapLegFixed,this).__init__(notional,schedule,currency,dayCounter)
        self.fixed_rate = fixed_rate
    
    def get_rate(self,startDate,endDate):
        return fixed_rate
    
    
class SwapLegFloating(SwapLeg):
    """ Represent a leg for a swap that has a floating amount with a spread
        added to it """
    def __init__(self,notional,schedule,currency,...,dayCounter=None):
        super(SwapLegFloating,this).__init__(notional,schedule,currency,dayCounter)
    
    def get_rate(self,startDate,endDate):
        pass
