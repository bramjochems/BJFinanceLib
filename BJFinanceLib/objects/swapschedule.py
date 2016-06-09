# -*- coding: utf-8 -*-
"""
Created on Tue May 31 22:52:26 2016

@author: Bram Jochems
"""
from abc import ABC, abstractmethod
import datetime
from numbers import Number
from BJFinanceLib.objects.cashflowschedule import CashflowSchedule

class SwapSchedule():
    """ Object representing the seperate periods for a swap(leg). Each period
        consists of a startdate, enddate and settlement date """
        
    def __init__(self,periods):
        """
        Initializes object.
        
        Argument
            - periods: Periods for the swap schedule. Can be specified as a list
                       of 1-tuples, 2-tuples or 3-tuples
                       1 tuples are assumed to be consecutive period starts and
                       ends (end of one period is star of the next) and
                       settlement dates are assumed to just be the end dates
                       2-tuples are assumed to be a period start and end each,
                       and settlement dates are assumed to be period end dates
                       3-tuples are assumed to be startdate, enddate and
                       settlement date
        """
        if len(periods) == 0:
            raise 'No periods provided'
        
        if isinstance(periods[0],datetime.date) or isinstance(periods[0],Number):
            periodLength = 1
        else:
            periodLength = len(periods[0])        
            
        if len(periods) == 1 and periodLength < 2:
            raise 'Single date is not valid for periods'
    
        if periodLength==1:
        # interpret single list of date as consecutive starts and ends
            starts = periods[0:-1]       
            ends = periods[1:]
            threeTuple = [(start,end,end) for (start,end) in zip(starts,ends)]            
      
        elif periodLength == 2:
        # interpret tuple as tuple of start and enddates with no setltement dates
            threeTuple is [(start,end,end) for (start,end) in periods]
        
        elif periodLength == 3:
        # interpret tuple as startdate, enddate and settlement date
            threeTuple = periods
            
        else:
            raise 'Periods specified by more than a 3-tuple are ambiguous'
        
        start, end, settle = zip(*sorted(threeTuple))
        self.period_startdates = list(start)
        self.period_enddates = list(end)
        self.period_settledates = list(settle)

    def __getitem__(self,index):
        """ Returns the tuple (startdate,enddate,settlement date) for index"""
        return (self.period_startdates(index),
                self.period_enddates(index),
                self.period_settledates(index))

    def unsettled_periods(self,referenceDate):
        """ Returns all periods whose settlement date is larger than
            referenceDate as a list of three tuples """
        return [(start,end,settle) for
                    (start,end,settle) in zip(self.period_startdates,
                                              self.period_enddates,
                                              self.period_settledates)
                    if settle > referenceDate]