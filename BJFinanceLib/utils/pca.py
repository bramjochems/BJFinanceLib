# -*- coding: utf-8 -*-
"""
Created on Mon May 23 18:05:20 2016

@author: M98I964
"""
import sklearn.decomposition as sd
from sklearn.preprocessing import StandardScaler

class PCA():
    """
    Class that performs principal component analysis. Uses sklearn internally,
    but simplifies to my typical usage.
    """
    def __init__(self,data,standardizeInputs=False,
                 varianceToKeep=None,
                 numberOfComponents=None):
        """
        Constructor
        
        Argumetns:
            - data to be reduced in dimension. Columns are the variables, rows
              the observations
            - standardizeInput: boolean that indicates whether input has to be
              normalized (demeaned and variance set to one). Optional, default
              is false.
            - varianceToKeep, optional boolean. If set to a value it determines
              the number of dimensions to retain by specifying the variance
              that the remaining variables should explain (as percentage of the
              total). If None (the default, it is ignored)
            - numberOfComponents: number of components to retain. If the
              varianceToKeep argument is set, this is ignored. If it isn't,
              and this argument isn't specified, all components are retained.
              If this argument is provided, it provides the number of components
              to keep.
              
        Returns:
           An object representing the PCA of the data. This has the following
           properties available:
               - explained_variance : total amount of variance explained by
                                      each component
               - explained_variance_ratio : normalized explained_variance
               - loadings : loadings of the PCA. These are the weights that
                            map a vector from the original basis to the pca
                            basis
               - scores: the representation of each point in the pca basis
          Finally, the object also has an method score availbable, see there
          for documentation.
        """
        self.__standardizeInputs = standardizeInputs
        self.__data = data if not standardizeInputs else StandardScaler().fit_transform(data)
        self.__pca = sd.PCA(varianceToKeep or numberOfComponents).fit(self.__data)

        self.explained_variance = self.__pca.explained_variance_
        self.explained_variance_ratio = self.__pca.explained_variance_ratio_
        self.loadings = self.__pca.components_
        self.scores = self.__pca.transform(self.__data)
        
    def score(self,data):
        """
        Scores new data on the pca basis determined by the original data
        
        Arguments:
            data : data, of the same dimension as the input data. If the
                   PCA object has standardizeInput set to True, it will
                   standardize this data too, otherwise it won't.
        Returns:
            Scores of the new data based on the PCA of the data this class was
            instantidated with.
        """
        std_data = data if not self.__standardizeInputs else StandardScaler().fit_transform(data)
        return self.__pca.transform(std_data)