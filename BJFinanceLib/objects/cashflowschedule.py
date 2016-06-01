# -*- coding: utf-8 -*-
"""
Object to represent a set of cash flows

@Author: Bram Jochems
"""

class CashflowSchedule():
    
    def __init__(self,dates,amounts,currency=None):
        """
        Initializes a cashflow schedule object.
        
        Arguments:
        dates - dates on which the cash flows in the schedule occur. Must have
                the same length as the amounts argument
        amounts - amounts of the cash flows on the dates. Must have same length
                  as the dates argument.
        currency - the currency in which the cash flows are specified. Optional
        """        
        
        if len(dates) != len(amounts):
            raise Exception("Dates and amounts must have same length")

        self.dates_with_amounts = dict(sorted(zip(dates,amounts)))
        """ List of all dates with associated cashflows """
        
        self.dates = self.dates_with_amounts.keys()
        """ List of all dates for which cashflows are avaiable"""
        
        self.currency = currency
        """ currency for the cashflow schedule """
           
    def amount_at_date(self,date):
        """ Retrieves the amount at a given date and 0 otherwise"""
        return self.dates_with_amounts.get(date,default=0.)