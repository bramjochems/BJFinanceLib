# -*- coding: utf-8 -*-

from BJFinanceLib.instruments.swapleg import SwapLegFixed, SwapLegFloating
from BJFinanceLib.objects.cashflowschedule import CashflowSchedule
from numbers import Number

class IRS():
    pass


class IRSFixedForFloat():

    @staticmethod
    def payerReceiver(payerReceiverFlag):
        if isinstance(payerReceiverFlag,str):
            payerReceiverFlag = payerReceiverFlag.lower()
            if payerReceiverFlag in ['p','pay','pay fixed','payer','fixed','fixed payer']:
                return 1.0
            elif payerReceiverFlag in ['r','rec','receiver','floating', 'pay floating','floating payer']:
                return -1.0
            else:
                raise Exception('Unhandled payer/receiver flag')
        
        elif isinstance(payerReceiverFlag,Number):
            if abs(abs(payerReceiverFlag)-1) < 1e-9:
                return round(payerReceiverFlag)
            else:
                raise Exception('Unhandled payer/receiver flag')
        else:
            raise Exception('Unhanded payer/receiver flag')
        
    """ A fixed for floating interest rate swap """
    def __init__(self,payerReceiverFlag, notional, schedule, floating_leg_spread=0,currency=None,dayCounterFixedLeg=None,dayCounterFloatingLeg=None):
        
        self.multiplier_payer_receiver = self.payerReceiver(payerReceiverFlag)   
        """ 1.0 for payer, -1.0 for receiver """

        self.notional = notional
        """ Notional for the swap """

        self.currency = currency        
        
        self.swapleg_fixed = SwapLegFixed(notional,
                                          schedule,
                                          fixed_rate,
                                          currency=currency,
                                          dayCounter=dayCounterFixedLeg)

        self.swapleg_floating = SwapLegFloating(notional,
                                                schedule,
                                                yieldFunction,
                                                spread=0,
                                                currency=currency,
                                                dayCounter=dayCounterFloatingLeg)
        
        self.discountFunction = ...        

    
    def update_floating_yield():
        pass
    
    def cashflow_schedule(self,referenceDate,selectLeg=None):
        """ Cashflow schedule for fixed swap

        Arguments:
            - referenceDate : date for which to get the cashflows.
            - selectLeg. Optional. If selectLeg = None, a netted cashflow
                         schedule for both legs is returned. If it is set to
                         'fixed' only the fixed leg schedule is returned. If
                         set to 'floating' only the floating leg schedule is
                         returned;
        """

        fixed_flows = self.swapleg_fixed.cashflow_schedule(referenceDate)
        floating_flows = self.cashflow_schedule(referenceDate)        

        pass
    
    def present_value(self,referenceDate):
        """ Present value of an IRS """
        discountFunction = ...
        return self.multiplier_payer_receiver * (
            self.swapleg_fixed.present_value(referenceDate,discountFunction) -
            self.swapleg_floating.present_value(referenceDate,discountFunction))