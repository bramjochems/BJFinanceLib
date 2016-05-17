# -*- coding: utf-8 -*-

import numpy as np

class OneDimensionalAntitheticRNG:
    """
    One dimensionsal random number generator that generates normally distributed
    randoms and uses antithetic variables. Note that this class using the numpy
    random class as underlying and setting the seed sets the seed for this class.
    This works fine for  the intended purposes of this class, but is dangerous
    in production level code where other parts of an application might depend
    on it too.
    """
    def __init__(self, numberOfPoints,seed=None):
        self.numberOfPoints = numberOfPoints
        self.replaceAntithetic = True
        self.next = np.zeros(numberOfPoints)
        self.hasSeed = not (seed==None)
        self.seed = seed
        if seed: np.random.seed(seed)
    
    def reset(self):
        """
        Resets the random number generator. Only works if it was initialized
        with a seed. Note that this resets the underlying seed in numpy!
        """
        if self.hasSeed:
            np.random.seed(self.seed)
        else:
            raise Exception("RNG initialized without seed, cannot reset")

    def getNormals(self):
        """
        Returns a new path of random variables
        """
        if self.replaceAntithetic:
            # If antithetic needs to be replaced, a new draw is necessary. The
            # result is returned directly and in a variable the next randoms
            # (which are just the old ones multiplied with -1) are stored
            tmp = np.random.randn(self.numberOfPoints)
            self.next = -tmp
            self.replaceAntithetic = False
            return tmp
        else:
            self.replaceAntithetic = True
            return self.next
            
class MultidimensionalRNG:
    """
    Multidimensional normally distributed RNG. Note that this class using the numpy
    random class as underlying and setting the seed sets the seed for this class.
    This works fine for  the intended purposes of this class, but is dangerous
    in production level code where other parts of an application might depend
    on it too.
    """
    def __init__(self,numberOfRows,numberOfColumns=1,seed=None):
        self.numberOfRows = numberOfRows
        self.numberOfColumns = numberOfColumns
        self.hasSeed = not (seed==None)
        self.seed = seed
        if seed: np.random.seed(seed)

    def reset(self):
        """
        Resets the random number generator. Only works if it was initialized
        with a seed. Note that this resets the underlying seed in numpy!
        """
        if self.hasSeed:
            np.random.seed(self.seed)
        else:
            raise Exception("RNG initialized without seed, cannot reset")       

    def getUncorrelatedNormals(self):
        """
        Returns a matrix with numberOfRows rows and numberOfColumns columns
        with uncorrelated normally distributed random draws
        """
        return np.random.randn(self.numberOfRows,self.numberOfColumns)
            