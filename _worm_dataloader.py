#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:14:29 2023

@author: Ranjiangshang Ran
"""

import xlrd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv as inv
from numpy import matmul as mmul

def loaddata(datapath, fps = 1e4, show = False, truncate_front = 0, truncate_end = 0):
    
    wb = xlrd.open_workbook(datapath) 
    sheet = wb.sheet_by_index(0) 
    x = sheet.col_values(colx = 0, start_rowx = 0, end_rowx = None)
    y = sheet.col_values(colx = 1, start_rowx = 0, end_rowx = None)

    x = np.asarray(x)
    y = np.asarray(y)

    data = np.array((x,y)).T
    data = data*1e-3 # [mm] to [m]
    
    # truncate the data
    data = data[0+truncate_front:-1-truncate_end,:]
    
    dt = 1/fps
    t = np.arange(len(data))*dt

    # use the first 20 point to do the linear fit to get velocity
    slt = 20 
    x_fit = data[:slt,0]
    y_fit = data[:slt,1]

    t_fit = np.arange(len(x_fit))*dt
    
    # make matrix multiplication work
    x_fit = np.array([x_fit]).T
    y_fit = np.array([y_fit]).T
    t_fit = np.array([t_fit]).T
    
    # assemble zeroth and first order matrix 
    t_fit = np.hstack((np.ones(x_fit.shape),t_fit))

    # Maximum likelihood estimation
    x_MLE = mmul(mmul(inv(mmul(t_fit.T,t_fit)),t_fit.T),x_fit)
    y_MLE = mmul(mmul(inv(mmul(t_fit.T,t_fit)),t_fit.T),y_fit)
    
    vec = np.hstack((x_MLE[1], y_MLE[1]))
    
    # set to be 0.6 m/s minimum
    V_mag = 0.6 
    xy_sum = V_mag**2. - vec[0]**2. - vec[1]**2.
    
    z_guess = np.sqrt(np.max([0.,xy_sum]))
    vec = np.hstack((vec, z_guess))
    
    print('u_x = %1.2f m/s' %vec[0])
    print('u_y = %1.2f m/s' %vec[1])
    print('u_z_guess = %1.2f m/s' %vec[2])
    
    if show:
        num_plot = 51
        t_plot = np.arange(num_plot)*dt
        x_plot = x_MLE[1]*t_plot + x_MLE[0]
        y_plot = y_MLE[1]*t_plot + y_MLE[0]
    
        plt.style.use(['default'])
        axfs = 12
        markersize = 2.25
        linewidth = 1.25

        plt.figure(figsize=(9,4))
        ax1 = plt.subplot(1,2,1)
        plt.plot(data[:,0]*1e3,data[:,1]*1e3,'ro', markersize = markersize)
        axxis = plt.gca()
        axxis.set_aspect('equal', adjustable='box')
        plt.grid(True)

        ax2 = plt.subplot(1,2,2)
        plt.plot(t*1e3,data[:,0]*1e3,'d', markersize = markersize)
        plt.plot(t*1e3,data[:,1]*1e3,'ro', markersize = markersize)
        
        plt.plot(t_plot*1e3,x_plot*1e3,'--', linewidth = linewidth, color = 'k')
        plt.plot(t_plot*1e3,y_plot*1e3,'--', linewidth = linewidth, color = 'k')

        ax1.set_xlabel(r'$x$ [mm]',fontsize=axfs)
        ax1.set_ylabel(r'$y$ [mm]',fontsize=axfs)
        ax1.tick_params(which = 'major', direction = 'in', length = 4, width = 1, labelsize = axfs)

        ax2.set_xlabel(r'$t$ [ms]',fontsize=axfs)
        ax2.set_ylabel(r'$x,y$ [mm]',fontsize=axfs)
        ax2.tick_params(which = 'major', direction = 'in', length = 4, width = 1, labelsize = axfs)
        plt.grid(True)
        plt.show()
        
    return t, data, vec