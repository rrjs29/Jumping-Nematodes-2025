#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 12:13 PM 2023
Modified on Mon July 21 11:20 PM 2025
@author: Ranjiangshang Ran
"""

import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import odeint
from numpy.linalg import norm as norm
from tqdm import tqdm
import pandas as pd

import seaborn as sns

from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

class RWMetropolis_joint:
    def __init__(self, log_p, rw_sigma):
        self.log_p = log_p
        self.rw_sigma = rw_sigma
        self.samples = []
        self.sigma_samples = []
        self.losshist = []
        
    def sample(self, x0, nIter = 3000):
        np.random.seed(1234)
        accepted = 0
        x = x0
        dim = x.shape[-1]
        log_sigma = 0 # initialize it at the mode log(simga) = 0
        logp, loss = self.log_p(x, log_sigma)
        
        for i in tqdm(range(nIter)):
            # June 4th, 2025, Ran: self.sigma is a standard deviation, not variance
            x_prop = x + self.rw_sigma * np.random.randn(dim)
            log_sigma_prop = log_sigma + self.rw_sigma*np.random.randn()

            logp_prop,loss_prop = self.log_p(x_prop, log_sigma_prop)
            
            alpha = min(1, np.exp(logp_prop - logp))
            if np.random.rand() < alpha:
                x = x_prop
                log_sigma = log_sigma_prop
                loss = loss_prop
                logp = logp_prop
                accepted += 1
            self.samples.append(x)
            self.sigma_samples.append(np.exp(log_sigma))
            self.losshist.append(loss)
        print('Acceptance rate: %.2f%%' % (100.0*accepted/nIter))
        
        losshist = np.asarray(self.losshist)
        plt.style.use(['default'])
        plt.figure(figsize=(6,4))
        plt.plot(losshist)
        plt.tick_params(which = 'major',direction = 'in',length=8, width=1, labelsize=16)
        plt.ylabel('Loss [mm]', fontsize=16)
        plt.xlabel('Iterations', fontsize=16)
        plt.grid(True)
        
class charge_fit_joint:
    def __init__(self, t, X, x0, init_guess, scales,
                 RW_sigma = 0.1, para_var = 1e1, burn_in = 500, a_f = 1e-3, phi_f = 100.):      
        self.t = t # [t], nt by 1 array
        self.X = X # [x,y], nt by 2 array, 2D trajectory
        self.N = X.shape[0]
        self.dim = X.shape[-1] # dimension of the observed data, here is 2
        self.x0 = x0 # [x0,y0], 1 by 2 array, initial position
        self.y_init = x0[1]
        
        self.scales = scales
        self.IG = init_guess
        
        # June 4th, 2025, Ran: I noted that RW_var here is actually RW_sigma,
        # meaning that it is random walk standard deviation, not variance.
        self.sampler = RWMetropolis_joint(self.log_posterior, RW_sigma)
        self.para_var = para_var
        self.burn_in = burn_in
        self.a_f = a_f
        self.phi_f = phi_f
        
    def log_prior(self, theta, log_sigma):
        # m = theta.shape[1] = 6 is the dimension of the phase space
        mu = 0
        tau_sq = 2.**2.
        
        log_prior = -0.5*theta.shape[1]*np.log(2*np.pi*self.para_var) - 0.5*norm(theta)**2./self.para_var
        log_prior += -0.5*np.log(2.*np.pi*tau_sq) - (log_sigma-mu)**2./(2.*tau_sq)
        
        return log_prior
    
    def log_likelihood(self, theta, log_sigma):
        # read theta to initial condition
        c1, c2, c3, c4 = theta.flatten()
        u_init = c3*self.scales[2] + self.IG[2] # [m/s], initial x velocity, initial guess is measured
        v_init = c4*self.scales[3] + self.IG[3] # [m/s], initial y velocity, initial guess is measured
        
        init_cond = [self.x0[0], self.x0[1], u_init, v_init]
        
        # exam residual
        sigma_sq = np.exp(2*log_sigma)
        mu = odeint(self.ode, init_cond, self.t, args = tuple([c1, c2]))
        
        X_pred = mu[:,0:2] # select the first two column [x,y] to check loss
        X = self.X
        Loss = norm(X - X_pred)*1e3
        
        # self.N = X.shape[0] is the number of frames in a trajectory
        # self.dim = 2 for x and y
        log_likelihood = -0.5*self.N*self.dim*np.log(2*np.pi*sigma_sq) - 0.5*Loss**2./sigma_sq
        
        return log_likelihood, Loss
    
    def log_posterior(self, theta, log_sigma):
        log_likelihood, Loss = self.log_likelihood(theta, log_sigma)
        log_posterior =  log_likelihood + self.log_prior(theta, log_sigma)
        return log_posterior, Loss
    
    def ode(self, x, t, c1, c2):
        # inferred parameters
        a_w = c1*self.scales[0] + self.IG[0] # [m], hydrodynamical radius of the worm, roughly 100~um
        Q_w = c2*self.scales[1] + self.IG[1] # [C], charge of the worm, roughly 0.1~pC 
        
        # fixed parameters
        pi = np.pi # [-]
        g0 = 9.81; # [m/s^2]
        eta = 1.849e-5; # [Pa.s], dynamic viscosity of air
        eps = 8.8541878128e-12; # [F/m], vacuum permittivity
        rho_w = 1e3; # [kg/m^3], density of the worm
#         a_w = 70e-6; # [m], hydrodynamical radius of the worm
        a_f = self.a_f; # [m], radius of the drop
#         phi_w = 240; # [V], voltage of the worm
        phi_f = self.phi_f; # [V], voltage of the fly
        
        # derived parameters from fixed parameters
        m_w = rho_w*pi*(12e-6)**2.0*500e-6; # [kg], mass of the worm, pi*r^2*L
#         Q_w = 4*pi*eps*a_w*phi_w; # [C], charge of the worm
        Q_f = 4*pi*eps*a_f*phi_f; # [C], charge of the fly
        
        # for v2 (droplet) data, the origin is defined to be the center of the droplet
        x0 = 0; # [m], fly x-position
        y0 = 0; # [m], fly y-position

        # eye-calibrated parameters, the mirror imgae fly position
        x2 = x0; # [m], fly x-position, same as x0
        y2 = 2*self.y_init - y0; # [m], fly y-position
    
        # variables that changes over time
        x1, y1, u1, v1 = x
        
        # derived variables that changes over time
        r1 = np.sqrt((x1-x0)**2.+(y1-y0)**2.);
        r2 = np.sqrt((x1-x2)**2.+(y1-y2)**2.);
        
        dxdt = [u1, v1,
                -6*pi*eta*a_w*u1/m_w - Q_w*Q_f/4/pi/eps*(x1-x0)/r1**3./m_w + Q_w*Q_f/4/pi/eps*(x1-x2)/r2**3./m_w,
                -6*pi*eta*a_w*v1/m_w - Q_w*Q_f/4/pi/eps*(y1-y0)/r1**3./m_w + Q_w*Q_f/4/pi/eps*(y1-y2)/r2**3./m_w - g0]
        
        return dxdt
    
    def show_iter2(self):
#         samples = np.asarray(self.sampler.samples)[self.burn_in:,:]
        samples = np.asarray(self.sampler.samples)[:,:2]
        iters = samples.shape[0]
        xticks = np.linspace(0,iters,6)

        plt.style.use(['default'])
        plt.figure(figsize=(6,10))
        
        ylabels = [r"$a_h\ [\mu$m]",r"$q$ [pC]", r"$u_0$ [m/s]",r"$v_0$ [m/s]",r"$\sigma$ [mm]"]

        plt.subplot(5,1,1)
        plt.plot(1e6*(self.scales[0]*samples[:,0,0]+self.IG[0]),color= 'tab:blue')
        plt.axhline(1e6*(self.scales[0]*np.mean(samples[self.burn_in:,0,0],0)+self.IG[0]),color='k', linestyle='--')
        plt.ylabel(ylabels[0], fontsize=16)
        plt.tick_params(which = 'major',direction = 'in',length=8, width=1, labelsize=16)
        plt.grid(True)
        # plt.yticks(np.array([100,110,120]))
        plt.xticks(xticks,[])
        
        plt.subplot(5,1,2)
        plt.plot(1e12*(self.scales[1]*samples[:,0,1]+self.IG[1]),color= 'tab:blue')
        plt.axhline(1e12*(self.scales[1]*np.mean(samples[self.burn_in:,0,1],0)+self.IG[1]),color='k', linestyle='--')
        plt.ylabel(ylabels[1], fontsize=16)
        plt.tick_params(which = 'major',direction = 'in',length=8, width=1, labelsize=16)
        plt.grid(True)
        plt.xticks(xticks,[])

        plt.subplot(5,1,3)
        plt.plot(1.*(self.scales[2]*samples[:,0,2]+self.IG[2]),color= 'tab:blue')
        plt.axhline(1.*(self.scales[2]*np.mean(samples[self.burn_in:,0,2],0)+self.IG[2]),color='k', linestyle='--')
        plt.ylabel(ylabels[2], fontsize=16)
        plt.tick_params(which = 'major',direction = 'in',length=8, width=1, labelsize=16)
        plt.grid(True)
        plt.xticks(xticks,[])
    
        plt.subplot(5,1,4)
        plt.plot(1.*(self.scales[3]*samples[:,0,3]+self.IG[3]),color= 'tab:blue')
        plt.axhline(1*(self.scales[3]*np.mean(samples[self.burn_in:,0,3],0)+self.IG[3]),color='k', linestyle='--')
        plt.ylabel(ylabels[3], fontsize=16)
        plt.tick_params(which = 'major',direction = 'in',length=8, width=1, labelsize=16)
        plt.grid(True)
        plt.xticks(xticks,[])

        sigma_samples = np.asarray(self.sampler.sigma_samples)
        
        plt.subplot(5,1,5)
        plt.plot(sigma_samples,color= 'tab:blue')
        plt.axhline(np.mean(sigma_samples[self.burn_in:]),color='k', linestyle='--')
        plt.ylabel(ylabels[4], fontsize=16)
        plt.tick_params(which = 'major',direction = 'in',length=8, width=1, labelsize=16)
        plt.grid(True)
        plt.xticks(xticks)
        plt.xlabel('Iterations', fontsize=16)
        plt.show()
            
    def show_mean_sol(self, show = True, long_t = 0):
        # get mean parameters from the sample
        samples = np.asarray(self.sampler.samples)[self.burn_in:,:]
        mean_samples = np.mean(samples,0)
        
        # read theta to initial condition
        c1, c2, c3, c4 = mean_samples.flatten()
        u_init = c3*self.scales[2] + self.IG[2] # [m/s], initial x velocity, initial guess is measured
        v_init = c4*self.scales[3] + self.IG[3] # [m/s], initial y velocity, initial guess is measured
        init_cond = [self.x0[0], self.x0[1], u_init, v_init]
        
        # exam loss in stress sigma
        mean_sol = odeint(self.ode, init_cond, self.t, args = tuple([c1, c2]))
        self.mean_sol = mean_sol
        
        # generate longer time for integration  
        dt = 1e-4
        t_longer = np.arange(len(self.t) + long_t) * dt
        long_sol = odeint(self.ode, init_cond, t_longer, args = tuple([c1, c2]))
        self.long_sol = long_sol
        
        if show:
            plt.style.use(['default'])
            axfs = 12
            markersize = 2.25
            linewidth = 1.25;
            
            plt.figure(figsize=(10,4))
            
            ax1 = plt.subplot(1,2,1)
            plt.title('Inferred Mean Solution',fontsize=14)
            plt.plot(self.X[:,0]*1e3,self.X[:,1]*1e3,'ro', markersize = markersize)
            plt.plot(self.mean_sol[:,0]*1e3,self.mean_sol[:,1]*1e3,'k-', markersize = markersize, linewidth = linewidth)
            axxis = plt.gca()
            axxis.set_aspect('equal', adjustable='box')
            plt.grid(True)

            ax2 = plt.subplot(1,2,2)
            plt.plot(self.t*1e3,self.X[:,0]*1e3,'o', markersize = markersize, color = 'tab:blue')
            plt.plot(self.t*1e3,self.X[:,1]*1e3,'o', markersize = markersize, color = 'tab:red')
            
            plt.plot(self.t*1e3,self.mean_sol[:,0]*1e3,'-', markersize = markersize, linewidth = linewidth, color = 'k')
            plt.plot(self.t*1e3,self.mean_sol[:,1]*1e3,'-', markersize = markersize, linewidth = linewidth, color = 'k')

            ax1.set_xlabel(r'$x$ [mm]',fontsize=axfs)
            ax1.set_ylabel(r'$y$ [mm]',fontsize=axfs)
            ax1.tick_params(which = 'major',direction = 'in',length=4, width=1, labelsize=axfs)

            ax2.set_xlabel(r'$t$ [ms]',fontsize=axfs)
            ax2.set_ylabel(r'$x,y$ [mm]',fontsize=axfs)
            ax2.tick_params(which = 'major',direction = 'in',length=4, width=1, labelsize=axfs)
            plt.grid(True)
            plt.show()
            
    def show_coeff(self, show_PDF = True, view = [30,-60]):
        samples = np.asarray(self.sampler.samples)[self.burn_in:,:]
        normalized_samples = samples
        
        for i in range(len(self.scales)):
            normalized_samples[:,0,i] = normalized_samples[:,0,i]*self.scales[i] + self.IG[i]
            
        # original values
        a_w_samples = normalized_samples[:,0,0]*1e6
        q_w_samples = normalized_samples[:,0,1]*1e12
        u_x_samples = normalized_samples[:,0,2]*1.
        u_y_samples = normalized_samples[:,0,3]*1.

        sigma_samples = np.asarray(self.sampler.sigma_samples)[self.burn_in:]
        
        # derived values
        U_samples = np.sqrt(normalized_samples[:,0,2]**2.+normalized_samples[:,0,3]**2.)
        theta_samples = np.arccos(u_y_samples/U_samples)/np.pi*180.
        
        coeff = np.vstack((q_w_samples,a_w_samples,u_x_samples,u_y_samples,sigma_samples))
        
        self.coeff = coeff
        
        print('q_w = %1.4f +/- %1.4f pC' %(np.mean(q_w_samples),np.std(q_w_samples)))
        print('a_w = %1.2f +/- %1.2f um' %(np.mean(a_w_samples),np.std(a_w_samples)))

        print('u_x = %1.4f +/- %1.4f m/s' %(np.mean(u_x_samples),np.std(u_x_samples)))
        print('u_y = %1.4f +/- %1.4f m/s' %(np.mean(u_y_samples),np.std(u_y_samples)))
        print('U = %1.4f +/- %1.4f m/s' %(np.mean(U_samples),np.std(U_samples)))
        print('theta = %1.2f +/- %1.2f deg' %(np.mean(theta_samples),np.std(theta_samples)))

        print('sigma = %1.4f +/- %1.4f mm' %(np.mean(sigma_samples),np.std(sigma_samples)))
        print('sigma = %1.4f +/- %1.4f pix' %(np.mean(sigma_samples)*80.,np.std(sigma_samples)*80.))
        
    def save_data(self, savepath = 'C:/Users/rran2/Jupyter/powersupplyFly_data/fit_results_v5/', savename = 'data.csv'):
        data = self.long_sol
        fullpath = savepath + savename
        savename2 = 'fit_' + savename
        
        fullpath2 = savepath + savename2
        
        meanss = np.mean(self.coeff, axis=1)
        stdss = np.std(self.coeff, axis=1)
        
        data2 = np.vstack((meanss,stdss))
        data2 = data2.T
        
        np.savetxt(fullpath, data, delimiter=',')
        np.savetxt(fullpath2, data2, delimiter=',')
        