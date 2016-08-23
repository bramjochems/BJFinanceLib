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
    
    def __init__(self, eq_spot, eq_vol_initial,
                 eq_vol_longterm, eq_mean_reversion_speed,
                 hw_a, hw_sigma_short_rate, hw_b, hw_sigma_long_rate,
                 rho_x_y, rho_S_sigma, rho_S_x, rho_S_y, rho_sigma_x, rho_sigma_y,
                 rates, sample_times):
        
        self.sample_times = preprocessSampleTimes(sample_times)
        self.__dt = list(zip(self.sample_times[:-1],self.sample_times[1:]))
        self.__rng = Randoms.MultidimensionalRNG(len(self.__dt),4)
        self.__initialize_initial_rates(rates)
        self.__initialize_cached_values(eq_spot, eq_vol_initial,
                 eq_vol_longterm, eq_mean_reversion_speed,
                 hw_a, hw_sigma_short_rate, hw_b, hw_sigma_long_rate,
                 rho_x_y, rho_S_sigma, rho_S_x, rho_S_y, rho_sigma_x, rho_sigma_y,
                 rates, sample_times)

    def __initialize_initial_rates(self,rates):
        """
        Initializes the initial rate curve by turning it into a function.
        Effectively this means that if the initial rates function wasn't in
        that form to start with, a lambda gets allcoted to inter-/extrapolate
        it
        """
        if hasattr(rates,'__call__'):
            self.__initial_rate = rates
        elif isinstance(rates,Number):
            self.__initial_rate = lambda t:rates
        else: # assume it's array/list of tuples:
            x,y = zip(*rates)
            f = ip.interp1d(x,y,fill_value="extrapolate")
            self.__initial_rate = lambda t:1*f(t)   


    def __initialize_cached_values(self, eq_spot, eq_vol_initial,
                 eq_vol_longterm, eq_mean_reversion_speed,
                 hw_a, hw_sigma_short_rate, hw_b, hw_sigma_long_rate,
                 rho_x_y, rho_S_sigma, rho_S_x, rho_S_y, rho_sigma_x, rho_sigma_y):
        """
        Initializes precomputed values and stores them in a dictionary (to
        avoid a host of private variables).
        """
        self.__precomputed = {}
        pass


    def __get_correlated_randoms(self):
        # first colum for vol, 2nd for x, third for y, forth for lnS
        pass 
    
    def __get_sigma_path(self,randoms):
        pass
    
    def __get_ou_path(self,precomputed_mean_rev, precomputed_stdev, randoms):
        pass
    
    def __get_x_path(self,randoms):
        pass
    
    def __get_y_path(self,randoms):
        pass
    
    def __get_lnS_path(self,xpath,ypath,sigmapath,randoms):
        pass
    
    def __calculate_r_path(self,xpath,ypath):
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
        randoms = self.__get_correlated_randoms();
        sigma_path = self.__get_sigma_path(randoms[:,0])
        x_path = self.__get_x_path(randoms[:,1])
        y_path = self.__get_y_path(randoms[:,2])
        r_path = self.__calculate_r_path(x_path,y_path)    
        lnS_path = self.__get_lnS_path(x_path,y_path,sigma_path,randoms[:,3])
        
        return np.column_stack([r_path,sigma_path,np.exp(lnS_path),x_path,y_path])