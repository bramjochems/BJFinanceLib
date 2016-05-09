# -*- coding: utf-8 -*-
"""
Nearest correlation matrix code found on th eweb. 
"""

#################   This Python code is designed by Yancheng Yuan at National University of Singapore to solve  ##
#################                          min 0.5*<X-Sigma, X-Sigma>
#################                          s.t. X_ii =b_i, i=1,2,...,n
#################                               X>=tau*I (symmetric and positive semi-definite)  ########
#################
#################                           based on the algorithm  in                   #################
#################                 ``A Quadratically Convergent Newton Method fo           r#################
#################                    Computing the Nearest Correlation Matrix             #################
#################                           By Houduo Qi and Defeng Sun                   #################
#################                     SIAM J. Matrix Anal. Appl. 28 (2006) 360--385.
#################
#################                       Last modified date: March 19, 2016                #################
#################                                                                       #################
#################  The  input arguments  Sigma, b>0, tau>=0, and tol (tolerance error)     #################
#################                                                                      #################
#################                                                                        #################
#################      For the correlation matrix problem, set b = np.ones((n,1))             #################
#################                                                                       #################
#################      For a positive definite matrix                                     #################
#################        set tau = 1.0e-5 for example                                    #################
#################        set tol = 1.0e-6 or lower if no very high accuracy required     #################
#################        If the algorithm terminates with the given tol, then it means    ##################
#################         the relative gap between the  objective function value of the obtained solution  ###
#################         and the objective function value of the global solution is no more than tol or  ###
#################         the violation of the feasibility is smaller than tol*(1+np.linalg.norm(b))    ############
#################
#################        The outputs include the optimal primal solution, the dual solution and others   ########
#################                 Diagonal Preconditioner is used                                  #################
#################
#################      Send your comments to hdqi@soton.ac.uk  or matsundf@nus.edu.sg              #################
#################
################# Warning:  Though the code works extremely well, it is your call to use it or not. #################

import numpy as np
import scipy

#Generate F(y) Compute the gradient
def _my_gradient(y_input, lamb, p_input, b_0, n):
    f = 0.0
    Fy = np.zeros((n, 1))
    p_input_copy = (p_input.copy()).transpose()
    for i in range(0,n):
        p_input_copy[i, :] = ((np.maximum(lamb[i],0).astype(float))**0.5)*p_input_copy[i, :]

    for i in range(0,n):
        Fy[i] = np.sum(p_input_copy[:, i]*p_input_copy[:, i])

    for i in range(0,n):
        f = f + np.square((np.maximum(lamb[i],0)))

    f = 0.5*f - np.dot(b_0.transpose(), y_input)

    return f, Fy


# use PCA to generate a primal feasible solution checked
def _my_pca(x_input, lamb, p_input, b_0, n):
    x_pca = x_input
    lamb = np.asarray(lamb)
    lp = lamb > 0
    r = lamb[lp].size
    if r == 0:
        x_pca = np.zeros((n, n))
    elif r == n:
        x_pca = x_input
    elif r<(n/2.0):
        lamb1 = lamb[lp].copy()
        lamb1 = lamb1.transpose()
        lamb1 = np.sqrt(lamb1.astype(float))
        P1 = p_input[:, 0:r].copy()
        if r>1:
            P1 = np.dot(P1,np.diagflat(lamb1))
            x_pca = np.dot(P1, P1.transpose())
        else:
            x_pca = np.dot(np.dot(np.square(lamb1), P1), P1.transpose())

    else:
        lamb2 = -lamb[r:n].copy()
        lamb2 = np.sqrt(lamb2.astype(float))
        p_2 = p_input[:, r:n]
        p_2 = np.dot(p_2,np.diagflat(lamb2))
        x_pca = x_pca + np.dot(p_2, p_2.transpose())

    # To make x_pca positive semidefinite with diagonal elements exactly b0
    d = x_pca.diagonal()
    d = d.reshape((d.size, 1))
    d = np.maximum(d, b_0.reshape(d.shape))
    x_pca = x_pca - np.diagflat(x_pca.diagonal()) + np.diagflat(d)
    d = d.astype(float)**(-0.5)
    d = d*((np.sqrt(b_0.astype(float))).reshape(d.shape))
    x_pca = x_pca*(np.dot(d, d.reshape(1, d.size)))

    return x_pca
# end of PCA

#To generate the first order difference of lambda
# To generate the first order essential part of d


def _my_omega_mat(p_input, lamb, n):
    idx_idp = np.where(lamb > 0)
    idx_idp = idx_idp[0]
    n = lamb.size
    r = idx_idp.size
    if r > 0:
        if r == n:
            omega_12 = np.ones((n, n))
        else:
            s = n - r
            dp = lamb[0:r].copy()
            dp = dp.reshape(dp.size, 1)
            dn = lamb[r:n].copy()
            dn = dn.reshape((dn.size, 1))
            omega_12 = np.dot(dp, np.ones((1, s)))
            omega_12 = omega_12/(np.dot(np.abs(dp), np.ones((1,s))) + np.dot(np.ones((r,1)), abs(dn.transpose())))
            omega_12 = omega_12.reshape((r, s))
    else:
        omega_12 = np.array([])
    return omega_12

# End of my_omega_mat


# To generate Jacobian
def _my_jacobian_matrix(x, omega_12, p_input, n):
    x_result = np.zeros((n,1))
    [r, s] = omega_12.shape
    if r > 0:
        hmat_1 = p_input[:, 0:r].copy()
        if r < n/2.0:
            i=0
            while i<n:
                hmat_1[i,:] = x[i]*hmat_1[i,:]
                i = i+1

            omega_12 = omega_12*(np.dot(hmat_1.transpose(), p_input[:, r:n]))
            hmat = np.dot(hmat_1.transpose(),np.dot(p_input[:, 0:r], p_input[:, 0:r].transpose()))
            hmat = hmat + np.dot(omega_12, p_input[:, r:n].transpose())
            hmat = np.vstack((hmat, np.dot(omega_12.transpose(), p_input[:, 0:r].transpose())))
            i = 0
            while i<n:
                x_result[i] = np.dot(p_input[i, :], hmat[:, i])
                x_result[i] = x_result[i] + 1.0e-10*x[i]
                i = i+1

        else:
            if r==n:
                x_result = 1.0e-10*x
            else:
                hmat_2 = p_input[:, r:n].copy()
                i=0
                while i<n:
                    hmat_2[i, :] = x[i]*hmat_2[i, :]
                    i = i+1

                omega_12 = np.ones((r, s)) - omega_12
                omega_12 = omega_12*(np.dot(p_input[:, 0:r].transpose(), hmat_2))
                hmat = np.dot(p_input[:, r:n].transpose(), hmat_2)
                hmat = np.dot(hmat, p_input[:, r:n].transpose())
                hmat = hmat + np.dot(omega_12.transpose(),p_input[:, 0:r].transpose())
                hmat = np.vstack((np.dot(omega_12, p_input[:, r:n].transpose()), hmat))
                i=0
                while i<n:
                    x_result[i] = np.dot(-p_input[i,:], hmat[:, i])
                    x_result[i] = x[i] + x_result[i] + 1.0e-10*x[i]
                    i = i+1

    return x_result

#end of Jacobian
# PCG Method
def _my_pre_cg(b, tol, maxit, c, omega_12, p_input, n):
    #Initializations
    r = b.copy()
    r = r.reshape(r.size, 1)
    c = c.reshape(c.size, 1)
    n2b = np.linalg.norm(b)
    tolb = tol*n2b
    p = np.zeros((n, 1))
    flag = 1
    iterk = 0
    relres = 1000
    # Precondition
    z = r/c
    rz_1 = np.dot(r.transpose(), z)
    rz_2 = 1
    d = z.copy()
    #d = d.reshape(z.shape)
    # CG Iteration
    for k in range(0, maxit):
        if k > 0:
            beta = rz_1/rz_2
            d = z + beta*d

        w = _my_jacobian_matrix(d, omega_12, p_input, n)
        denom = np.dot(d.transpose(), w)
        iterk = k+1
        relres = np.linalg.norm(r)/n2b
        if denom <= 0:
            #ss = 0 # don't know the usage, check the paper
            p = d/np.linalg.norm(d)
            break
        else:
            alpha = rz_1/denom
            p = p + alpha*d
            r = r - alpha*w

        z = r/c
        if np.linalg.norm(r)<=tolb: #exit if hmat p = b solved in relative error tolerance
            iterk = k+1
            relres = np.linalg.norm(r)/n2b
            flag = 0
            break

        rz_2 = rz_1
        rz_1 = np.dot(r.transpose(), z)

    return p, flag, relres, iterk

# end of pre_cg

#to generate the diagonal preconditioner


def _my_precond_matrix(omega_12, p_input, n):
    [r, s] = omega_12.shape
    c = np.ones((n, 1))
    if r > 0:
        if r < n/2.0:
            hmat = (p_input.copy()).transpose()
            hmat = hmat*hmat
            hmat_12 = np.dot(hmat[0:r, :].transpose(), omega_12)
            d = np.ones((r, 1))
            for i in range(0, n):
                c_temp = np.dot(d.transpose(), hmat[0:r, i])
                c_temp = c_temp*hmat[0:r, i]
                c[i] = np.sum(c_temp)
                c[i] = c[i] + 2.0*np.dot(hmat_12[i, :], hmat[r:n, i])
                if c[i] < 1.0e-8:
                    c[i] = 1.0e-8

        else:
            if r < n:
                hmat = (p_input.copy()).transpose()
                hmat = hmat*hmat
                omega_12 = np.ones((r,s)) - omega_12
                hmat_12 = np.dot(omega_12, hmat[r:n, :])
                d = np.ones((s, 1))
                dd = np.ones((n, 1))

                for i in range(0, n):
                    c_temp = np.dot(d.transpose(), hmat[r:n, i])
                    c[i] = np.sum(c_temp*hmat[r:n, i])
                    c[i] = c[i] + 2.0*np.dot(hmat[0:r, i].transpose(), hmat_12[:, i])
                    alpha = np.sum(hmat[:, i])
                    c[i] = alpha*np.dot(hmat[:, i].transpose(), dd) - c[i]
                    if c[i] < 1.0e-8:
                        c[i] = 1.0e-8

    return c

# end of precond_matrix 


# my_issorted()

def _my_issorted(x_input, flag):
    n = x_input.size
    tf_value = False
    if n < 2:
        tf_value = True
    else:
        if flag == 1:
            i = 0
            while i < n-1:
                if x_input[i] <= x_input[i+1]:
                    i = i+1
                else:
                    break

            if i == n-1:
                tf_value = True
            elif i < n-1:
                tf_value = False

        elif flag == -1:
            i = n-1
            while i > 0:
                if x_input[i] <= x_input[i-1]:
                    i = i-1
                else:
                    break

            if i == 0:
                tf_value = True
            elif i > 0:
                tf_value = False

    return tf_value

# end of my_issorted()


def _my_mexeig(x_input):
    [n, m] = x_input.shape
    [lamb, p_x] = np.linalg.eigh(x_input)
    #lamb = lamb.reshape((lamb.size, 1))
    p_x = p_x.real
    lamb = lamb.real
    if _my_issorted(lamb, 1):
        lamb = lamb[::-1]
        p_x = np.fliplr(p_x)
    elif _my_issorted(lamb, -1):
        return p_x, lamb
    else:
        idx = np.argsort(-lamb)
       #lamb_old = lamb   # add for debug
        lamb = lamb[idx]
        #p_x_old = p_x   add for debug
        p_x = p_x[:, idx]

    lamb = lamb.reshape((n, 1))
    p_x = p_x.reshape((n, n))

    return p_x, lamb

# end of my_mymexeig()


# begin of the main function


def _my_correlationmatrix(g_input, b_input=None, tau=None, tol=None):
    [n, m] = g_input.shape
    g_input = g_input.copy()
    g_input = (g_input + g_input.transpose())/2.0
    b_g = np.ones((n, 1))
    error_tol = 1.0e-6
    if b_input is None:
        tau = 0
    elif tau is None:
        b_g = b_input.copy()
        tau = 0
    elif tol is None:
        b_g = b_input.copy() - tau*np.ones((n, 1))
        g_input = g_input - tau*np.eye(n, n)
    else:
        b_g = b_input.copy() - tau*np.ones((n, 1))
        g_input = g_input - tau*np.eye(n, n)
        error_tol = np.maximum(1.0e-12, tol)

    res_b = np.zeros((300,1))
    norm_b0 = np.linalg.norm(b_g)
    y = np.zeros((n, 1))
    f_y = np.zeros((n, 1))
    k=0
    f_eval = 0
    iter_whole = 200
    iter_inner = 20 # maximum number of line search in Newton method
    maxit = 200 # maximum number of iterations in PCG
    iterk = 0

    tol_cg = 1.0e-2 # relative accuracy for CGs
    sigma_1 = 1.0e-4
    x0 = y.copy()

    c = np.ones((n, 1))
    d = np.zeros((n, 1))
    val_g = np.sum((g_input.astype(float))*(g_input.astype(float)))
    val_g = val_g*0.5
    x_result = g_input + np.diagflat(y)
    x_result = (x_result + x_result.transpose())/2.0
    [p_x, lamb] = _my_mexeig(x_result)
    [f_0, f_y] = _my_gradient(y, lamb, p_x, b_g, n)
    initial_f = val_g - f_0
    x_result = _my_pca(x_result, lamb, p_x, b_g, n)
    val_obj = np.sum(((x_result - g_input)*(x_result - g_input)))/2.0
    gap = (val_obj - initial_f)/(1.0 + np.abs(initial_f) + np.abs(val_obj))
    f = f_0.copy()
    f_eval = f_eval + 1
    b_input = b_g - f_y
    norm_b = np.linalg.norm(b_input)
    omega_12 = _my_omega_mat(p_x, lamb, n)
    x0 = y.copy()

    while np.abs(gap) > error_tol and norm_b/(1+norm_b0) > error_tol and k < iter_whole:
        c = _my_precond_matrix(omega_12, p_x, n)

        [d, flag, relres, iterk] = _my_pre_cg(b_input, tol_cg, maxit, c, omega_12, p_x, n)
        slope = np.dot((f_y - b_g).transpose(), d)

        y = (x0 + d).copy()
        x_result = g_input + np.diagflat(y)
        x_result = (x_result + x_result.transpose())/2.0
        [p_x, lamb] = _my_mexeig(x_result)
        [f, f_y] = _my_gradient(y, lamb, p_x, b_g, n)

        k_inner = 0
        while k_inner <= iter_inner and f > f_0 + sigma_1*(np.power(0.5, k_inner))*slope + 1.0e-6:
            k_inner = k_inner + 1
            y = x0 + (np.power(0.5, k_inner))*d
            x_result = g_input + np.diagflat(y)
            x_result = (x_result + x_result.transpose())/2.0
            [p_x, lamb] = _my_mexeig(x_result)
            [f, f_y] = _my_gradient(y, lamb, p_x, b_g, n)

        f_eval = f_eval + k_inner + 1
        x0 = y.copy()
        f_0 = f.copy()
        val_dual = val_g - f_0
        x_result = _my_pca(x_result, lamb, p_x, b_g, n)
        val_obj = np.sum((x_result - g_input)*(x_result - g_input))/2.0
        gap = (val_obj - val_dual)/(1 + np.abs(val_dual) + np.abs(val_obj))

        b_input = b_g - f_y
        norm_b = np.linalg.norm(b_input)
        #rel_norm_b = norm_b/(1+norm_b0)
        res_b[k] = norm_b
        k = k + 1
        omega_12 = _my_omega_mat(p_x, lamb, n)

    #position_rank = np.maximum(lamb, 0)>0
    x_result = x_result + tau*(np.eye(n))
    return x_result, y

# end of the main function

def nearest_correlation_matrix(matrix,tau=1e-5,tol=1e-6):
    [n,_]= matrix.shape
    [x_res,_] = _my_correlationmatrix(matrix,np.ones((n,1)), tau, tol)
    return x_res 