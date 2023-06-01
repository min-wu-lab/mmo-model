#!/usr/bin/env python
# coding: utf-8

# # Mixed-mode oscillation 'teacup' model
# 
# Edited: May 17, 2023 </br>
# By: XJ
# 
# 
# Based on: <div class="csl-entry">Hastings, A., &#38; Powell, T. (1991). Chaos in a three-species food chain. <i>Ecology</i>, <i>72</i>(3), 896–903. https://doi.org/10.2307/1940591</div>
# 
# $X \rightarrow{} Y \rightarrow{} Z$
# 
# $Y \rightarrow{} D$
# 
# $Z \rightarrow{} E$
# 
# $f(\sigma) = \frac{\sigma}{1+\sigma}$
# 
# $\frac{dx}{dt}= x(1-x) - \frac{a_1 x}{1 + b_1 x} y$
# 
# $\frac{dy}{dt}= \frac{a_1 x}{1 + b1 x} y - \frac{a_2 y}{1 + b_2 y} z - d_1 y$
# 
# $\frac{dz}{dt}= \frac{a_2 y}{1 + b_2 y} z - d_2 z$

# In[12]:


# ------------  Import Packages ------------ 
import os
import matplotlib.pyplot as plt
import imageio
from numpy import *
import pylab as p
from scipy import integrate
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform

# ------------  Define arrow object ------------ 
class Arrow3D(FancyArrowPatch):
    """
    Class for patching arrow objects on 3D trajectory plots.
    """
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs) 
    
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D, 'arrow3D', _arrow3D)


# ## Mixed-mode oscillations

# In[50]:


savefigure = 1 # 1 for save, 0 otherwise
figname = 'MMO_1^3' # figure file name
fz = 36 # fontsize

# ------------  Define parameters and functions ------------

a1 = 5.0
a2 = 0.1
b1 = 3.0
b2 = 2.0
d1 = 0.4
d2 = 0.01

def f1(u):
    return a1 * u / (1 + b1 * u)

def f2(u):
    return a2 * u / (1 + b2 * u)

def dX_dt(X, t=0):
    """ 
    return [dx/dt, dy/dt, dz/dt]
    """
    return array([ X[0] * (1 - X[0]) - f1(X[0]) * X[1],
                   f1(X[0]) * X[1]  - f2(X[1]) * X[2] - d1 * X[1],
                   f2(X[1]) * X[2] - d2 * X[2] ])

# ------------  Integrate ODEs ------------
X0 = array([1, 1, 10]) # initial conditions
t = linspace(0, 6500, 100000) # time array
idt = 100000/6500 # inverse stepsize
X = integrate.odeint(dX_dt, X0, t) # integrate

# ------------  Plot trajectory ------------
ax = plt.figure(figsize=(16,12)).add_subplot(projection='3d')

# select only one cycle to plot
t0, tend = int(6100*idt), int(6236*idt)
ax.plot(X[t0:tend,0], X[t0:tend,1], X[t0:tend,2], color='black', lw=4)

# patch arrows
tp = array([int(t0+100+i*300) for i in range(7)])
for ti in tp:
    dX = dX_dt(X[ti,:],t[ti])
    ax.arrow3D(X[ti,0], X[ti,1], X[ti,2],
           dX[0]/10, dX[1]/10, dX[2]/10,
           mutation_scale=40,
           fc='black') 

ax.set_xlabel("x", fontsize=fz)
ax.set_ylabel("y", fontsize=fz)
ax.set_zlabel("z", fontsize=fz)

ax.set_xticks([0,1])
ax.set_yticks([0.4])
ax.set_zticks([8,10])
ax.tick_params(axis='both', which='major', labelsize=fz)

ax.view_init(20, -170)
ax.grid(False)
plt.tight_layout()
if savefigure: plt.savefig(figname+'phase-space.eps')
plt.show()
plt.close()

# ------------  Plot time series ------------
# plot the last 500 au timepoints to ensure convergence
f1 = p.figure()
p.figure(figsize=(10,6))
p.plot(t-6000, X[:,1], '-', color='black', label='y', lw=3)
p.xlabel('Time', fontsize=fz)
p.ylabel('y', fontsize=fz)
p.xticks([0,500], fontsize=fz)
p.yticks([0,1], fontsize=fz)
p.xlim((0,500))
p.ylim((0,1))
plt.tight_layout()
if savefigure: plt.savefig(figname+'time-series.eps')
plt.show()
plt.close()


# ## Simple oscillations

# In[47]:


savefigure = 1 # 1 for save, 0 otherwise
figname = 'SO' # figure file name
fz = 36 # fontsize

# ------------  Define parameters and functions ------------

a1 = 7.0
a2 = 0.1
b1 = 3.0
b2 = 2.0
d1 = 0.4
d2 = 0.01

def f1(u):
    return a1 * u / (1 + b1 * u)

def f2(u):
    return a2 * u / (1 + b2 * u)

def dX_dt(X, t=0):
    """ 
    return [dx/dt, dy/dt, dz/dt]
    """
    return array([ X[0] * (1 - X[0]) - f1(X[0]) * X[1],
                   f1(X[0]) * X[1]  - f2(X[1]) * X[2] - d1 * X[1],
                   f2(X[1]) * X[2] - d2 * X[2] ])

# ------------  Integrate ODEs ------------
X0 = array([1, 1, 10]) # initial conditions
t = linspace(0, 6500, 100000) # time array
idt = 100000/6500 # inverse stepsize
X = integrate.odeint(dX_dt, X0, t) # integrate

# ------------  Plot trajectory ------------
ax = plt.figure(figsize=(16,12)).add_subplot(projection='3d')

# select only one cycle to plot
t0, tend = int(6100*idt), int(6236*idt)
ax.plot(X[t0:tend,0], X[t0:tend,1], X[t0:tend,2], color='black', lw=4)

# patch arrows
# tp = array([int(t0+100+i*300) for i in range(7)])
tp = t0 + np.array([0, 230, 280])
for ti in tp:
    dX = dX_dt(X[ti,:],t[ti])
    ax.arrow3D(X[ti,0], X[ti,1], X[ti,2],
           dX[0]/10, dX[1]/10, dX[2]/10,
           mutation_scale=40,
           fc='black') 

ax.set_xlabel("x", fontsize=fz)
ax.set_ylabel("y", fontsize=fz)
ax.set_zlabel("z", fontsize=fz)

ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_zticks([0,10])
ax.tick_params(axis='both', which='major', labelsize=fz)

ax.view_init(20, -170)
ax.grid(False)
plt.tight_layout()
if savefigure: plt.savefig(figname+'phase-space.eps')
plt.show()
plt.close()

# ------------  Plot time series ------------
# plot the last 500 au timepoints to ensure convergence
f1 = p.figure()
p.figure(figsize=(10,6))
p.plot(t-6000, X[:,1], '-', color='black', label='y', lw=3)
p.xlabel('Time', fontsize=fz)
p.ylabel('y', fontsize=fz)
p.xticks([0,500], fontsize=fz)
p.yticks([0,1], fontsize=fz)
p.xlim((0,500))
p.ylim((0,1))
plt.tight_layout()
if savefigure: plt.savefig(figname+'time-series.eps')
plt.show()
plt.close()


# ## Fixed point

# In[46]:


savefigure = 1 # 1 for save, 0 otherwise
figname = 'FP' # figure file name
fz = 36 # fontsize

# ------------  Define parameters and functions ------------

a1 = 1.0
a2 = 0.1
b1 = 3.0
b2 = 2.0
d1 = 0.4
d2 = 0.01

def f1(u):
    return a1 * u / (1 + b1 * u)

def f2(u):
    return a2 * u / (1 + b2 * u)

def dX_dt(X, t=0):
    """ 
    return [dx/dt, dy/dt, dz/dt]
    """
    return array([ X[0] * (1 - X[0]) - f1(X[0]) * X[1],
                   f1(X[0]) * X[1]  - f2(X[1]) * X[2] - d1 * X[1],
                   f2(X[1]) * X[2] - d2 * X[2] ])

# ------------  Integrate ODEs ------------
X0 = array([1, 1, 10]) # initial conditions
t = linspace(0, 6500, 100000) # time array
idt = 100000/6500 # inverse stepsize
X = integrate.odeint(dX_dt, X0, t) # integrate

# ------------  Plot trajectory ------------
ax = plt.figure(figsize=(16,12)).add_subplot(projection='3d')

# t0, tend = int(100*idt), int(6500*idt)
# ax.plot(X[t0:tend,0], X[t0:tend,1], X[t0:tend,2], color='black', lw=4)

X0 = array([[3, 1, 1],
           [0.1, 1, 0.5],
           [5, 1, 1.5]])
for i in range(np.shape(X0)[0]):
    X = integrate.odeint(dX_dt, X0[i], t)
    ax.plot(X[:,0], X[:,1], X[:,2], color='black', lw=4)
    ax.scatter(X[-1,0], X[-1,1], X[-1,2],s=300, c='black', marker='o')
    tp = array([20, 80, 800])
    for ti in tp:
        dX = dX_dt(X[ti,:],t[ti])
        ax.arrow3D(X[ti,0], X[ti,1], X[ti,2],
               dX[0]/10, dX[1]/10, dX[2]/10,
               mutation_scale=40,
               fc='black') 

ax.set_xlabel("x", fontsize=fz)
ax.set_ylabel("y", fontsize=fz)
ax.set_zlabel("z", fontsize=fz)

ax.set_xticks([0,5])
ax.set_yticks([0,1])
ax.set_zticks([0,2])
ax.tick_params(axis='both', which='major', labelsize=fz)

ax.view_init(20, -170)
ax.grid(False)
plt.tight_layout()
if savefigure: plt.savefig(figname+'phase-space.eps')
plt.show()
plt.close()

# ------------  Plot time series ------------
# plot the last 500 au timepoints to ensure convergence
f1 = p.figure()
p.figure(figsize=(10,6))
p.plot(t-6000, X[:,1], '-', color='black', label='y', lw=3)
p.xlabel('Time', fontsize=fz)
p.ylabel('y', fontsize=fz)
p.xticks([0,500], fontsize=fz)
p.yticks([0,1], fontsize=fz)
p.xlim((0,500))
p.ylim((0,1))
plt.tight_layout()
if savefigure: plt.savefig(figname+'time-series.eps')
plt.show()
plt.close()


# ## Mixed-mode osc of higher order - more chaotic

# In[38]:


savefigure = 1 # 1 for save, 0 otherwise
figname = 'MMO_higher-order' # figure file name
fz = 36 # fontsize

# ------------  Define parameters and functions ------------

a1 = 5.0
a2 = 0.1
b1 = 5.0
b2 = 2.0
d1 = 0.4
d2 = 0.01

def f1(u):
    return a1 * u / (1 + b1 * u)

def f2(u):
    return a2 * u / (1 + b2 * u)

def dX_dt(X, t=0):
    """ 
    return [dx/dt, dy/dt, dz/dt]
    """
    return array([ X[0] * (1 - X[0]) - f1(X[0]) * X[1],
                   f1(X[0]) * X[1]  - f2(X[1]) * X[2] - d1 * X[1],
                   f2(X[1]) * X[2] - d2 * X[2] ])

# ------------  Integrate ODEs ------------
X0 = array([1, 1, 10]) # initial conditions
t = linspace(0, 6500, 100000) # time array
idt = 100000/6500 # inverse stepsize
X = integrate.odeint(dX_dt, X0, t) # integrate

# ------------  Plot trajectory ------------
ax = plt.figure(figsize=(16,12)).add_subplot(projection='3d')

t0, tend = int(5500*idt), int(6500*idt)
ax.plot(X[t0:tend,0], X[t0:tend,1], X[t0:tend,2], color='black', lw=4)

# patch arrows
# tp = array([int(t0+100+i*300) for i in range(7)])
# for ti in tp:
#     dX = dX_dt(X[ti,:],t[ti])
#     ax.arrow3D(X[ti,0], X[ti,1], X[ti,2],
#            dX[0]/10, dX[1]/10, dX[2]/10,
#            mutation_scale=40,
#            fc='black') 

ax.set_xlabel("x", fontsize=fz)
ax.set_ylabel("y", fontsize=fz)
ax.set_zlabel("z", fontsize=fz)

ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_zticks([0,10])
ax.tick_params(axis='both', which='major', labelsize=fz)

ax.view_init(20, -170)
ax.grid(False)
plt.tight_layout()
if savefigure: plt.savefig(figname+'phase-space.eps')
plt.show()
plt.close()

# ------------  Plot time series ------------
# plot the last 500 au timepoints to ensure convergence
f1 = p.figure()
p.figure(figsize=(10,6))
p.plot(t-5500, X[:,1], '-', color='black', label='y', lw=3)
p.xlabel('Time', fontsize=fz)
p.ylabel('y', fontsize=fz)
p.xticks([0,1000], fontsize=fz)
p.yticks([0,1], fontsize=fz)
p.xlim((0,1000))
p.ylim((0,1))
plt.tight_layout()
if savefigure: plt.savefig(figname+'time-series.eps')
plt.show()
plt.close()


# In[ ]:




