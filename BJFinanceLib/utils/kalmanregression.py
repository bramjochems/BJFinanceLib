# -*- coding: utf-8 -*-
from numbers import Number
import numpy as np
from BJFinanceLib.utils.kalman import KalmanFilter

class KalmanRegression():
    """
    Class that performs online regression where parameters are updated based on
    a Kalman Filter.
    """
    
    
    
    def __init__(self,X,y,
                 add_intercept=False,
                 burn_in_period=None,
                 initial_state_mean=None,
                 observation_covariance=None,
                 initial_state_covariance=None,
                 transition_covariance=None ):
        """
        Initializes the class.
        
        Arguments
        X : observations with each row a distinct observation. Can be either a
            np array or a pandas dataframe.
            
        y : target values for the regression. Must be (effectively) one-
            dimensional.
            
        add_intercept: Optional boolean, default=False. If true, an intercept
                       is added to the model as a last regression coefficient.
                       
        burn_in_period: Optional int that specifies the number of observations
                        left out for initial training of the algorithm. If not
                        specified, no initial calibration on training data is
                        done (which makes error statistics a bit worse). The
                        burn_in_period just uses a regular regression for burn-
                        in, not some online method.
                 
        initial_state_mean: Initial mean for the states. Optional, starting
                            from zero if not specified. If a burn_in_period is
                            specfied, this paramter gets ignored.
                 
        observation_covariance: Noise inherent to the y-values. Scalar.
                                Optional, if not specified, an estimated value
                                is used that might be completely off.
                                
        initial_state_covariance: Optional input for state covariances to start
                                  from. The covariance estimates get updated
                                  over time. Ignored if a burn-in period is set.
                                  The initial state covariance determines how
                                  initially the 

        transition_covariance: How much our parameters can vary over time. Optional.
                               Square matrix of size equal to the number of
                               regressors (remember to add one for the
                               intercept if an intercept is used) or a scalar.
.
        """
        # Process arrays into uniform input format
        X_internal = np.reshape(X.copy(), (len(X),-1)) # Always 2d
        y_internal = np.reshape(y.copy(), (len(y),1)) # Always 2D array but only single column 
        nobs,nvars = np.shape(X_internal)        
        if add_intercept:
            X_internal = np.hstack((X_internal,np.ones((len(X_internal),1))))
            nvars = nvars+1

        X_internal = np.expand_dims(X_internal,axis=1)
        
        # Some input validation            
        if (len(X_internal) != len(y_internal)):
            raise Exception("Different number of observations and independent variables")
            
        if burn_in_period:
            raise Exception("Burn-in period not implemented yet")
            #To implement, make sure taht initial_state_mean, initial_sate_cov,
            #observation_covariance and transition_covariance are filled.
        else:
            if not initial_state_mean:
                initial_state_mean = [0 for i in range(nvars)]
                
            if not observation_covariance:
                observation_covariance = np.std(y_internal) / 5 # Assume 20% of noise of y is inherent to y
                
            if initial_state_covariance:
                if isinstance(initial_state_covariance,Number):
                    initial_state_cov = initial_state_covariance*np.ones((nvars,nvars))
                else:
                    initial_state_cov = initial_state_covariance
            else:
                initial_state_cov = np.ones((nvars,nvars))
                        
            if not transition_covariance:
                transition_covariance = 0.01 * np.eye(nvars)
            elif isinstance(transition_covariance,Number):
                transition_covariance *= np.eye(nvars)
                            
        self.__kalman = KalmanFilter(n_dim_obs=1,
                                     n_dim_state=nvars,
                                     initial_state_mean=initial_state_mean,
                                     initial_state_covariance=initial_state_cov,
                                     transition_matrices=np.eye(nvars),
                                     observation_matrices=X_internal,
                                     observation_covariance=observation_covariance,
                                     transition_covariance=transition_covariance)
        """ The internal kalman filter class that's used """
        
        state_means,state_covs = self.__kalman.filter(y_internal) #do the actual filtering

        self.rolling_coefficients = state_means
        """
        Provides the regression coefficients as an array where the rows are the
        various points available and the columns contain the regression weights.
        """
        
        self.rolling_coefficient_covariances = state_covs        
        """
        Covariance matrix for the regression paramters as estimated by the
        Kalman filter.
        """
    