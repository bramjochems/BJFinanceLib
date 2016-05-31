# -*- coding: utf-8 -*-
"""
Created on Tue May 31 22:52:26 2016

@author: Bram Jochems
"""

import BJFinanceLib.analytical.rates

repository = BJFinanceLib.analytical.rates.YieldCurveRepository

class IRS():
    """ Class to represent IRS trades """
    pass

def clean_price(irs):
    """ Determines clean price for an IRS """
    ...
    
def dirty_price(irs):
    """ Determines dirty price for an IRS """    
    
def bpv_profile(irs,
                buckets=(['3M','6M','1Y','2Y','3Y','4Y','5Y','7Y','10Y',
                          '15Y','20Y','25Y','30Y'])):
    """ Determines a bpv profile for an irs """
    ...
    
