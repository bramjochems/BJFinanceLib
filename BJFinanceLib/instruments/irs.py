# -*- coding: utf-8 -*-

from BJFinanceLib.instruments.swapleg import SwapLegFixed, SwapLegFloating
from BJFinanceLib.objects.cashflowschedule import CashflowSchedule

class IRS():
    pass

class IRSFixedForFloat():
    """ A fixed for floating interest rate swap """
    def __init__(currency=None):
        
        self.multiplier_payer_receiver = ...        
        
        self.swapleg_fixed = ...

        self.swapleg_floating = ...
        
        self.discountFunction = ...        
        
        pass
    
    def update_floating_yield():
        pass
    
    def cashflow_schedule(self,referenceDate):
        fixed_flows = self.swapleg_fixed.cashflow_schedule(referenceDate)
        floating_flows = self.cashflow_schedule(referenceDate)        

        pass
    
    def present_value(self,referenceDate):
        pass