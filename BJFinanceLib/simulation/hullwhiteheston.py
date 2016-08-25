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
from math import exp, sqrt, log
from numbers import Number
import numpy as np
import scipy.interpolate as ip
from scipy.linalg import cholesky
import BJFinanceLib.simulation.rng as Randoms
from BJFinanceLib.simulation.utils import preprocessSampleTimes, validateNumberParam

class HullWhiteHestonGenerator():
    """
    Class to generate paths for a combined rates + equity process. For the
    rates process a G2++ model is used and a Heston process for the equities.
    Based on http://dare.uva.nl/cgi/arno/show.cgi?fid=481155 (thesis from Laura
    Khune)
    """
    def __init__(self, eq_spot, eq_vol_initial,
                 eq_vol_longterm, eq_mean_reversion_speed, eq_vol_of_vol,
                 hw_a, hw_sigma_short_rate, hw_b, hw_sigma_long_rate,
                 rho_x_y, rho_S_sigma, rho_S_x, rho_S_y, rho_sigma_x, rho_sigma_y,
                 rates, sample_times):
        """
        Constructor. Doc to do.
        """
        self.sample_times = preprocessSampleTimes(sample_times)
        self._dt = [t-s for (s,t) in list(zip(self.sample_times[:-1],self.sample_times[1:]))]
        self._rng = Randoms.MultidimensionalRNG(len(self._dt),4)
        self._initialize_initial_rates(rates)
        self._initialize_cached_values( eq_spot, eq_vol_initial,
                 eq_vol_longterm, eq_mean_reversion_speed, eq_vol_of_vol,
                 hw_a, hw_sigma_short_rate, hw_b, hw_sigma_long_rate,
                 rho_x_y, rho_S_sigma, rho_S_x, rho_S_y, rho_sigma_x, rho_sigma_y)


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
        try:
            validateNumberParam(value,lb,ub)
            self._inputs[description] = value
        except:
            raise Exception('Error while setting validating and setting ' + 
                            description + ' to value ' + str(value))

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
        self._initialize_sigma_params()
        self._initialize_xy_params()
        self._initialize_s_path_params()
        self._initialize_phi()
        
        
    def _initialize_sigma_params(self):
        """"
        Precomputes some parameters needed in the volatility path generation
        """        
        self._precomputed['sigma_params'] = {}
        smean = self._inputs['heston_vol_long_term']
        omega = self._inputs['heston_vol_of_vol']
        epsilon = self._inputs['heston_vol_meanreversion_speed']
        exp_edt = [exp(-epsilon*dt) for dt in self._dt]
        self._precomputed['sigma_params']['s2_constant'] = [smean*omega**2*(1-edt)**2/(2*epsilon) for edt in exp_edt]
        self._precomputed['sigma_params']['s2_factor'] = [ omega**2*edt*(1-edt)/epsilon for edt in exp_edt]    
        self._precomputed['sigma_params']['m_constant'] = [smean*(1-edt) for edt in exp_edt]
        self._precomputed['sigma_params']['m_factor'] = [edt for edt in exp_edt]
     
     
    def _initialize_xy_params(self):
        """
        Intializes parameters for calculation of x and y paths for G2++ model
        """
        for name in ['x','y']:
            self._precomputed[name+'path'] = {} 
            self._precomputed[name+'path']['mean_rev'] = [exp(-self._inputs['hullwhite_'+name +'_meanreversion']*dt) for dt in self._dt]
            self._precomputed[name+'path']['stdev'] = [sqrt((self._inputs['hullwhite_'+name +'_vol']**2)*
                                                            (1-exp(-2*self._inputs['hullwhite_'+ name +'_meanreversion']*dt))/
                                                            (2*self._inputs['hullwhite_'+name+'_meanreversion'])) for dt in self._dt]
     
    def _initialize_phi(self):
        """ Initializes the phi(t) of the G2++ model """
        s = self._inputs['hullwhite_x_vol']
        e = self._inputs['hullwhite_y_vol']
        a = self._inputs['hullwhite_x_meanreversion']
        b = self._inputs['hullwhite_y_meanreversion']
        rho_xy = self._inputs['correl_x_y']
        self._precomputed['phi_vector'] = [self._initial_rate(t) +
                                           0.5*(s/a*(1-exp(-a*t)))**2 +
                                           0.5*(e/b*(1-exp(-b*t)))**2 +
                                           rho_xy*s*e/(a*b)*(1-exp(-a*t))*(1-exp(-b*t))
                                                for t in self.sample_times] 
     
    def _initialize_s_path_params(self):
        """
        Precomputes parameters needed when generating the stock path
        """
        self._precomputed['stockpath'] = {}
        self._precomputed['stockpath']['lnS0'] = log(self._inputs['equity_spot'])
        self._precomputed['stockpath']['integral_phi'] = np.array( 
             [self._initial_rate(t)*t - self._initial_rate(s)*s +
              0.5*(self._vt_helper(0,t)-self._vt_helper(0,s)) for 
                   (s,t) in list(zip(self.sample_times[:-1],self.sample_times[1:]))])
    
    def _vt_helper(self,t,T):
        if T <= t:
            return 0
        else:
            a = self._inputs['hullwhite_x_meanreversion']
            b = self._inputs['hullwhite_y_meanreversion']
            s = self._inputs['hullwhite_x_vol']
            eta = self._inputs['hullwhite_y_vol']
            rho = self._inputs['correl_x_y']
            eat = exp(-a*(T-t))
            ebt = exp(-b*(T-t))
    
            part1 = (s/a)**2  * (T-t + 2*eat/a - 0.5*exp(-2*a*(T-t))/a - 1.5/a)
            part2 = (eta/b)**2 * (T-t + 2*ebt/b - 0.5*exp(-2*b*(T-t))/b - 1.5/b)
            part3 = 2*rho*s*eta/(a*b)*(T-t + (eat-1)/a + (ebt-1)/b - (eat*ebt-1)/(a+b))
            return part1+part2+part3
        
    
    def _determine_cholesky_decomposition(self):
        """
        Cholesky decomposition for correlation matrix. Based on analytic
        expressions rather than numerical approximation. Assumes as order for
        the randoms: sigma, x, y, lnS
        """
        input_mat = np.array([[1,                                self._inputs['correl_vol_x'],   self._inputs['correl_vol_y'],   self._inputs['correl_stock_vol']],
                              [self._inputs['correl_vol_x'],     1,                              self._inputs['correl_x_y'],     self._inputs['correl_stock_x']],
                              [self._inputs['correl_vol_y'],     self._inputs['correl_x_y'],     1,                              self._inputs['correl_stock_y']],
                              [self._inputs['correl_stock_vol'], self._inputs['correl_stock_x'], self._inputs['correl_stock_y'], 1                             ]])
        return cholesky(input_mat)


    def _get_correlated_randoms(self,randoms_to_use=None):
        """
        Returns a matrix of samples from a correlated standard normal
        distribution. First colum for vol, 2nd for x, third for y,
        fourth for lnS
        """
        if randoms_to_use == None:
            randoms_to_use = self._rng.get_uncorrelated_normals()
        elif np.shape(randoms_to_use) != (len(self._dt),4):
            raise Exception('Random variables incorrectly dimensioned')
        return np.dot(randoms_to_use,self._precomputed['cholesky'])
    
    
    def _get_sigma_path(self,randoms):
        """
        Generates a path for volatility for the Heston process using Andersen's
        QE scheme.
        """
        res = np.zeros(len(self.sample_times))
        res[0] = self._inputs['heston_vol_initial']
        uvec = np.random.uniform(size=len(self._dt))
        for (cntr,dt) in enumerate(self._dt):
            s2 = res[cntr]*self._precomputed['sigma_params']['s2_factor'][cntr] + self._precomputed['sigma_params']['s2_constant'][cntr]
            m = res[cntr]*self._precomputed['sigma_params']['m_factor'][cntr] + self._precomputed['sigma_params']['m_constant'][cntr]           
            phi = s2/(m**2)
            if phi <= 1.5:
                b2 = 2/phi - 1 + sqrt(2/phi)*sqrt(2/phi-1)
                a = m / (1+b2)
                b = sqrt(b2)
                res[cntr+1] = a*(b+randoms[cntr])**2
            else:
                p = (phi-1)/(phi+1)
                beta = (1-p)/m
                u = uvec[cntr]
                if u < p:
                    res[cntr+1] = 0
                else:
                    res[cntr+1] = log((1-p)/(1-u))/beta
        return res
 
   
    def _get_ou_path(self,precomputed_mean_rev, precomputed_stdev, randoms):
        """ Helper function for generating paths for x and y """
        res = np.zeros(len(self.sample_times))
        res[0] = 0

        for (cntr,(m,s,random)) in enumerate(zip(precomputed_mean_rev,precomputed_stdev,randoms)):
            res[cntr+1] = res[cntr]*m + s*random
        return res
   
   
    def _get_x_path(self,randoms):
        """ Generate path of x variable in G2++ model """
        return self._get_ou_path(self._precomputed['xpath']['mean_rev'],
                                 self._precomputed['xpath']['stdev'],
                                 randoms)
  
  
    def _get_y_path(self,randoms):
        """ Generate path of y variable in G2++ model """
        return self._get_ou_path(self._precomputed['ypath']['mean_rev'],
                                 self._precomputed['ypath']['stdev'],
                                 randoms)
    
    
    def _get_lnS_path(self,xpath,ypath,sigmapath,randoms):
        """
        Calculates the path for the log of the stock price
        """
        dt = np.array(self._dt)
        integral_ru_helper = (0.5*(xpath[1:]+xpath[:-1]+ypath[1:]+ypath[:-1])*dt + 
                              self._precomputed['stockpath']['integral_phi'])
        integral_sudu_helper = np.sqrt(0.5*(sigmapath[1:]+sigmapath[:-1])*dt)
        
        e = self._inputs['heston_vol_meanreversion_speed']        
        w = self._inputs['heston_vol_of_vol']
        sm= self._inputs['heston_vol_long_term']
        rss = self._inputs['correl_stock_vol']
        K0 = [-rss*e*sm*dt/w for dt in self._dt]
        K1 = [0.5*dt*(rss*e/w-0.5) - rss/w for dt in self._dt]
        K2 = [0.5*dt*(rss*e/w-0.5) + rss/w for dt in self._dt]
        C42 = self._precomputed['cholesky'][1,3]
        C43 = self._precomputed['cholesky'][2,3]
        C44 = self._precomputed['cholesky'][3,3]
        randhelper = (C42*randoms[:,1]+C43*randoms[:,2]+C44*randoms[:,3])*integral_sudu_helper
        drifthelper = K0 + K1*sigmapath[:-1] + K2*sigmapath[1:]
        
        res = np.zeros(len(self.sample_times))
        res[0]  = self._precomputed['stockpath']['lnS0']
        res[1:] = integral_ru_helper + drifthelper + randhelper   
        return np.cumsum(res)
    
    
    def _calculate_r_path(self,xpath,ypath):
        return xpath + ypath + self._precomputed['phi_vector']
    
    
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
        Based on QE scheme for Heston (see Andersen, 2008)
        """
        randoms = self._get_correlated_randoms();
        sigma_path = self._get_sigma_path(randoms[:,0])
        x_path = self._get_x_path(randoms[:,1])
        y_path = self._get_y_path(randoms[:,2])
        r_path = self._calculate_r_path(x_path,y_path)    
        lnS_path = self._get_lnS_path(x_path,y_path,sigma_path,randoms)
        return np.column_stack([r_path,np.exp(lnS_path),sigma_path,x_path,y_path])