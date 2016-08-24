# -*- coding: utf-8 -*-
"""
d ln[S(t)] = (r(t)-0.5 sigma(t))dt + sqrt(sigma(t)) dW_S
d sigma(t) = epsilon(sigma_mean-sigma(t))dt + omega sqrt(sigma(t)) dW_Sigma
d x(t) = -a x(t) dt + sigma_rate_short dW_x
d y(t) = -b y(t) dt + sigma_rate_long dW_y
r(t) = x(t) + y(t) + phi(t)

<dW_x,dW_y) = rho_xy dt
<dW_S,dW_sigma> = rho_Ssigma dt
<dW_S,dW_x> = rho_Sx dt
<dW_S,dW_y> = rho_Sy dt
<dW_sigma,dW_x> = rho_sigmax dt
<dW_sigam,dW_y> = rho_sigmay dt

PSEUDO-CODE:
- Retrieve choleksy decomposition of matrix
- For each simulation:
     - draw correlated random normals
     - generate path for simga(t)
     - generate path for x(t)
     - generate path for y(t)
     - retrieve r(t) path
     - retrieve ln S(t) path
"""
from math import exp, sqrt
from numbers import Number
import numpy as np
import scipy.interpolate as ip
import BJFinanceLib.simulation.rng as Randoms
from BJFinanceLib.simulation.utils import preprocessSampleTimes, validateNumberParam

class HullWhiteHestonGenerator():
    """
    Class to generate paths for a combined rates + equity process. For the
    rates process a G2++ model is used and a Heston process for the equities.
    """
    def __init__(self, eq_spot, eq_vol_initial,
                 eq_vol_longterm, eq_mean_reversion_speed, eq_vol_of_vol,
                 hw_a, hw_sigma_short_rate, hw_b, hw_sigma_long_rate,
                 rho_x_y, rho_S_sigma, rho_S_x, rho_S_y, rho_sigma_x, rho_sigma_y,
                 rates, sample_times):
        
        self.sample_times = preprocessSampleTimes(sample_times)
        self._dt = list(zip(self.sample_times[:-1],self.sample_times[1:]))
        self._rng = Randoms.MultidimensionalRNG(len(self._dt),4)
        self._initialize_initial_rates(rates)
        self._initialize_cached_values(eq_spot, eq_vol_initial,
                 eq_vol_longterm, eq_mean_reversion_speed,
                 hw_a, hw_sigma_short_rate, hw_b, hw_sigma_long_rate,
                 rho_x_y, rho_S_sigma, rho_S_x, rho_S_y, rho_sigma_x, rho_sigma_y,
                 rates, sample_times)


    def _initialize_initial_rates(self,rates):
        """
        Initializes the initial rate curve by turning it into a function.
        Effectively this means that if the initial rates function wasn't in
        that form to start with, a lambda gets allcoted to inter-/extrapolate
        it
        """
        if hasattr(rates,'_call_'):
            self._initial_rate = rates
        elif isinstance(rates,Number):
            self._initial_rate = lambda t:rates
        else: # assume it's array/list of tuples:
            x,y = zip(*rates)
            f = ip.interp1d(x,y,fill_value="extrapolate")
            self._initial_rate = lambda t:1*f(t)   


    def _check_and_assign_input_parameter(self,description,value,
                                           lb=-np.inf,ub=np.inf):
        validateNumberParam(value,lb,ub)
        self._inputs['description'] = value
        

    def _initialize_cached_values(self, eq_spot, eq_vol_initial,
                 eq_vol_longterm, eq_mean_reversion_speed, eq_vol_of_vol,
                 hw_a, hw_sigma_short_rate, hw_b, hw_sigma_long_rate,
                 rho_x_y, rho_S_sigma, rho_S_x, rho_S_y, rho_sigma_x, rho_sigma_y):
        """
        Initializes precomputed values and stores them in a dictionary (to
        avoid a host of private variables).
        """
        # deal with input parameters
        self._inputs = {}
        param_values = [('equity_spot',eq_spot,0.001,np.inf),
                        ('heston_vol_initial', eq_vol_initial,0.001,25),
                        ('heston_vol_long_term', eq_vol_longterm, 0.001, 25),
                        ('heston_vol_meanreversion_speed', eq_mean_reversion_speed,0,1.1/max(self._dt)),
                        ('heston_vol_of_vol',eq_vol_of_vol,0,100),
                        ('hullwhite_x_meanreversion',hw_a,0,1.1/max(self._dt)),
                        ('hullwhite_x_vol',hw_sigma_short_rate,0,1),
                        ('hullwhite_y_meanreversion',hw_b,0,1.1/max(self._dt)),
                        ('hullwhite_y_vol',hw_sigma_long_rate,0,1),
                        ('correl_x_y',rho_x_y,-1,1),
                        ('correl_stock_x',rho_S_x,-1,1),
                        ('correl_stock_y',rho_S_y,-1,1),
                        ('correl_stock_vol',rho_S_sigma,-1,1),
                        ('correl_vol_x',rho_sigma_x,-1,1),
                        ('correl_vol_y',rho_sigma_y,-1,1)]                  
        for (description,value,lb,ub) in param_values:
            self._check_and_assign_input_parameter(description,value,lb,ub)
        
        # deal with precomputation of other values.
        self._precomputed = {}
        self._precomputed['cholesky'] = self._determine_cholesky_decomposition()

    def _determine_cholesky_decomposition(self):
        """
        Cholesky decomposition for correlation matrix. Based on analytic
        expressions rather than numerical approximation. Assumes as order for
        the randoms: sigma, x, y, lnS
        """
        chol = np.zeros((4,4))
        chol[0,0] = 1
        chol[0,1] = self._inputs['correl_vol_x']
        chol[0,2] = self._inputs['correl_vol_y']
        chol[0,3] = self._inputs['correl_stock_vol']
        helper = sqrt(1-self._inputs['correl_vol_x']**2)
        chol[1,1] = helper
        chol[1,2] = (self._inputs['correl_x_y'] - 
                     self._inputs['correl_vol_x']*self._inputs['correl_vol_y'])/helper
        chol[1,3] = (self._inputs['correl_stock_x'] - self._inputs['correl_stock_vol']*
                                                       self._inputs['correl_vol_x'])/helper
        helper3 = -1+self._inputs['correl_vol_x']**2
        helper2 = (self._inputs['correl_vol_x']**2 +
                   self._inputs['correl_vol_y']**2 +
                        -2*self._inputs['correl_vol_x']*
                           self._inputs['correl_vol_y']*
                           self._inputs['correl_x_y'])
                           
        c33 = sqrt(1 + helper2/helper3)
        #different than paper, think there is a mistake there!
        helper4 =(-self._inputs['correl_stock_vol']*self._inputs['correl_vol_y'] +
                   self._inputs['correl_vol_x']*self._inputs['correl_vol_y']*
                                              self._inputs['correl_stock_x'] +
                   self._inputs['correl_stock_vol']*self._inputs['correl_vol_x']*
                                                  self._inputs['correl_x_y'] -
                   self._inputs['correl_stock_x']*self._inputs['correl_x_y'] +
                   self._inputs['correl_stock_y'] -
                   self._inputs['correl_stock_y']*self._inputs['correl_vol_x']**2)
        c43 = helper4/sqrt(helper3*(helper2+helper3))
        c44 = sqrt(1 + self._inputs['correl_stock_vol']**2 +
                   (self._inputs['correl_stock_x'] - self._inputs['correl_stock_vol']*
                                                      self._inputs['correl_vol_x'])**2/helper3 + 
                   (helper4**2)/(helper3*(helper2+helper3)))        
        chol[2,2] = c33
        chol[2,3] = c43
        chol[3,3] = c44
        return chol

    def _get_correlated_randoms(self,randoms_to_use=None):
        """
        Returns a matrix of samples from a correlated standard normal
        distribution. First colum for vol, 2nd for x, third for y,
        fourth for lnS
        """
        if randoms_to_use == None:
            randoms_to_use = self._rng.GetUncorrelatedNormals()
        elif np.shape(randoms_to_use) != (len(self._dt),4):
            raise Exception('Random variables incorrectly dimensioned')
        return np.dot(randoms_to_use,self._precomputed['cholesky'])

    
    def _get_sigma_path(self,randoms):
        pass
    
    def _get_ou_path(self,precomputed_mean_rev, precomputed_stdev, randoms):
        pass
    
    def _get_x_path(self,randoms):
        pass
    
    def _get_y_path(self,randoms):
        pass
    
    def _get_lnS_path(self,xpath,ypath,sigmapath,randoms):
        pass
    
    def _calculate_r_path(self,xpath,ypath):
        pass
    
    def get_path(self):
        """
        Returns a single path from the simulation. Output is a nx5 array. The
        number of rows is the number of evaluation dates (including t=0). The
        columns are (in order):
            - short rate
            - equity price
            - volatility path
            - x part of the G2++ process
            - y part of the G2++ process
        """
        randoms = self._get_correlated_randoms();
        sigma_path = self._get_sigma_path(randoms[:,0])
        x_path = self._get_x_path(randoms[:,1])
        y_path = self._get_y_path(randoms[:,2])
        r_path = self._calculate_r_path(x_path,y_path)    
        lnS_path = self._get_lnS_path(x_path,y_path,sigma_path,randoms[:,3])
        return np.column_stack([r_path,sigma_path,np.exp(lnS_path),x_path,y_path])