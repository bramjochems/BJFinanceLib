# -*- coding: utf-8 -*-
from numbers import Number
from numpy import inf
def preprocessSampleTimes(sampleTimes):
    """
    Helper function that takes sample tiime points for a monte carlo routine as
    input and 'sanitizes' it.
    """
    if isinstance(sampleTimes,Number): sampleTimes = [sampleTimes]
    return [0] + [t for t in sorted(sampleTimes) if t > 0]
    
def validateNumberParam(singleParam, min_value=-inf, max_value=inf):
    if (not isinstance(singleParam,Number) or 
        singleParam < min_value or
        singleParam > max_value):
        raise Exception("Invalid parameter")   