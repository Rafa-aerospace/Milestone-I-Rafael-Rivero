"""
Created on Fri Sep  9 16:13:53 2022

@author: Rafael Rivero de Nicolás
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib import rc # LaTeX tipography
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rc('text', usetex=True); plt.rc('font', family='serif')

import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)

# %% Functions Definitions

# It only depends on the physics problem 
def F(X, t):
    '''
    Parameters
    ----------
    X : Array
        State vector of the system in instant t.
    t : Float
        Time instant in which F is being evaluated.

    Returns
    -------
    Array
        First derivate of the tate vector. dU/dt = F(U,t).

    '''
    F1 = X[2]
    
    F2 = X[3]
    
    F3 = -X[0]/(X[0]**2 + X[1]**2)**(3/2)
    
    F4 = -X[1]/(X[0]**2 + X[1]**2)**(3/2)

    return np.array([F1, F2, F3, F4])


def RK4(X, t, dt):
    '''
    Parameters
    ----------
    X : Array
        State vector of the system in instant t.
    t : Array
        Time instant in which F is being evaluated.
    dt : Float
        Time step used during the simulation, also called Δt.

    Returns
    -------
    Array
        Second term for computing U_(n+1), state vector in instant t+dt.
        U_(n+1) = U_(n) + ( k1 + 2*k2 + 2*k3 + k4 ) / 6.

    '''
    
    k1 = F( X, t )
    
    k2 = F( X + dt * k1/2, t + dt/2 )
    
    k3 = F( X + dt * k2/2, t + dt/2 )
    
    k4 = F( X + dt *k3,    t + dt   )
    
    return ( k1 + 2*k2 + 2*k3 + k4 ) / 6


def Crank_Nicolson(X, t, dt):
    '''
    Parameters
    ----------
    X : Array
        State vector of the system in instant t.
    t : Array
        Time instant in which F is being evaluated.
    dt : Float
        Time step used during the simulation, also called Δt.

    Returns
    -------
    Array
        U is the state vector of the system in instant t+dt. 
        It is also the vector that satisfies: U = X + dt/2 * ( F(X,t) + F(U,t+dt) ) .

    '''
    
    def  Crank_Nicolson_Operator(U_n1):
        return  U_n1 - X - dt/2 * ( F(X,t) + F(U_n1,t+dt) )
    
    U = fsolve(Crank_Nicolson_Operator, X)
    
    return U

# %% Initialitation

r_0 = np.array([1, 0]) # Initial position

v_0 = np.array([0, 1]) # Initial velocity

U_0 = np.hstack((r_0,v_0)) #U_0 = np.array([r_0[0], r_0[1], v_0[0], v_0[1]])

print('Initial State Vector: U_0 = ', U_0)

TIME = 20

Delta_t = [0.2, 0.1, 0.01, 0.001]                # Δt for different simulations

U_Euler = {}      # Initialitation of U (State vector for each simulation at each instant time)
U_RK4 = {}
U_CrankNic = {}

for key in Delta_t:     # Initialitation of zeros matrices for each scheme

    U_Euler[key] =    np.zeros((len(U_0), int(TIME/key)+1))
    
    U_RK4[key] =      np.zeros((len(U_0), int(TIME/key)+1))
    
    U_CrankNic[key] = np.zeros((len(U_0), int(TIME/key)+1))

    U_Euler[key][:,0] = U_0
    
    U_RK4[key][:,0] = U_0
    
    U_CrankNic[key][:,0] = U_0


# %% Schemes Application

for dt in Delta_t:
    
    t = float(0)
    it = 1 # Number of iteration
    
    while t < TIME:
        
        U_Euler[dt][:,it] = U_Euler[dt][:,it-1] + F( U_Euler[dt][:,it-1], t ) * dt
        
        U_RK4[dt][:,it]   = U_RK4[dt][:,it-1]   + RK4( U_RK4[dt][:,it-1], t, dt ) * dt
        
        U_CrankNic[dt][:,it] = Crank_Nicolson(U_CrankNic[dt][:,it-1], t, dt)
        
        t = round( t + dt, 3 )  # To avoid round-off error
        it = it + 1


# %% Plotting

x = np.linspace(-1,1,1000)
y = np.sqrt(1-x**2)

colours = ['purple', 'orange', 'blue', 'red']

fig, ax = plt.subplots(1,1, figsize=(11,11), constrained_layout='true')
ax.set_xlim(-1.85,1.85)
ax.set_ylim(-1.85,1.85)
ax.set_title('Numeric Scheme: Euler', fontsize=20)
ax.grid()
ax.set_xlabel(r'$x$',fontsize=20)
ax.set_ylabel(r'$y$',fontsize=20)
for i in range(len(Delta_t)):
    
    ax.plot( U_Euler[Delta_t[i]][0,:], U_Euler[Delta_t[i]][1,:], c=colours[i], label=r'$\Delta t$ = '+str(Delta_t[i]))

ax.plot(x,y,'k'); ax.plot(x,-y,'k')
ax.plot(0,0,'k-o', markersize=12)
ax.legend(loc=0, fancybox=False, edgecolor="black", ncol = 1, fontsize=16)
plt.show()


fig, ax = plt.subplots(1,1, figsize=(11,11), constrained_layout='true')
# ax.set_xlim(0.25,0.5)
# ax.set_ylim(0.75,1)
ax.set_xlim(-1.1,1.1)
ax.set_ylim(-1.1,1.1)
ax.set_title('Numeric Scheme: Runge Kutta 4th order', fontsize=20)
ax.grid()
ax.set_xlabel(r'$x$',fontsize=20)
ax.set_ylabel(r'$y$',fontsize=20)
for i in range(len(Delta_t)):
    
    ax.plot( U_RK4[Delta_t[i]][0,:], U_RK4[Delta_t[i]][1,:], c=colours[i], label=r'$\Delta t$ = '+str(Delta_t[i]))

ax.plot(x,y,'k'); ax.plot(x,-y,'k')
ax.plot(0,0,'k-o', markersize=12)
ax.legend(loc=0, fancybox=False, edgecolor="black", ncol = 1, fontsize=16)
plt.show()

# fig, ax = plt.subplots(1,1, figsize=(11,11), constrained_layout='true')
# ax.set_xlim(0.25,0.5)
# ax.set_ylim(0.75,1)
# # ax.set_xlim(-1.1,1.1)
# # ax.set_ylim(-1.1,1.1)
# ax.set_title('Numeric Scheme: Runge Kutta 4th order')
# ax.grid()
# ax.set_xlabel(r'$x$',fontsize=20)
# ax.set_ylabel(r'$y$',fontsize=20)
# for i in range(len(Delta_t)):
    
#     ax.plot( U_RK4[Delta_t[i]][0,:], U_RK4[Delta_t[i]][1,:], c=colours[i])

# ax.plot(x,y,'k'); ax.plot(x,-y,'k')
# ax.plot(0,0,'k-o', markersize=12)
# plt.show()


fig, ax = plt.subplots(1,1, figsize=(11,11), constrained_layout='true')
# ax.set_xlim(0.25,0.5)
# ax.set_ylim(0.75,1)
ax.set_xlim(-1.1,1.1)
ax.set_ylim(-1.1,1.1)
ax.set_title('Numeric Scheme: Crank-Nicolson (Implicit)',fontsize=20)
ax.grid()
ax.set_xlabel(r'$x$',fontsize=20)
ax.set_ylabel(r'$y$',fontsize=20)
for i in range(len(Delta_t)):
    
    ax.plot( U_CrankNic[Delta_t[i]][0,:], U_CrankNic[Delta_t[i]][1,:], c=colours[i], label=r'$\Delta t$ = '+str(Delta_t[i]))

ax.plot(x,y,'k'); ax.plot(x,-y,'k')
ax.plot(0,0,'k-o', markersize=12)
ax.legend(loc=0, fancybox=False, edgecolor="black", ncol = 1, fontsize=16)
plt.show()


# %%
fig, axes = plt.subplots(3,1, figsize=(18,10), constrained_layout='true')

ax = axes[0]
ax.set_xlim(0,t)
ax.set_title('Conservation of kinetic energy in different schemes with'+r' $\Delta t =$ '+str(Delta_t[-1]) ,fontsize=20)
ax.grid()
ax.set_xlabel(r'$t$',fontsize=20)
ax.set_ylabel(r'$(\dot{x}^2 + \dot{y}^2)^{\frac{1}{2}}$',fontsize=20)
ax.plot( np.linspace(0,t,len(U_CrankNic[Delta_t[-1]][0,:])), np.sqrt(U_CrankNic[Delta_t[-1]][2,:]**2 + U_CrankNic[Delta_t[-1]][3,:]**2) , c='black', label='Crank-Nicolson')
ax.plot( np.linspace(0,t,len(U_Euler[Delta_t[-1]][0,:])), np.sqrt(U_Euler[Delta_t[-1]][2,:]**2 + U_Euler[Delta_t[-1]][3,:]**2) , c='blue', label='Euler')
ax.plot( np.linspace(0,t,len(U_RK4[Delta_t[-1]][0,:])), np.sqrt(U_RK4[Delta_t[-1]][2,:]**2 + U_RK4[Delta_t[-1]][3,:]**2) , c='red', label='Runge Kutta 4th')
ax.legend(loc=0, fancybox=False, edgecolor="black", ncol = 1, fontsize=16)

ax = axes[1]
ax.set_xlim(0,t)
ax.set_title('Conservation of kinetic energy with Runge Kutta 4th order scheme',fontsize=20)
ax.grid()
ax.set_xlabel(r'$t$',fontsize=20)
ax.set_ylabel(r'$(\dot{x}^2 + \dot{y}^2)^{\frac{1}{2}}$',fontsize=20)
ax.plot( np.linspace(0,t,len(U_RK4[Delta_t[0]][0,:])), np.sqrt(U_RK4[Delta_t[0]][2,:]**2 + U_RK4[Delta_t[0]][3,:]**2) , c='green', label=r'$\Delta t =$ '+str(Delta_t[0]))
ax.plot( np.linspace(0,t,len(U_RK4[Delta_t[1]][0,:])), np.sqrt(U_RK4[Delta_t[1]][2,:]**2 + U_RK4[Delta_t[1]][3,:]**2) , c='purple', label=r'$\Delta t =$ '+str(Delta_t[1]))
ax.plot( np.linspace(0,t,len(U_RK4[Delta_t[-1]][0,:])), np.sqrt(U_RK4[Delta_t[-1]][2,:]**2 + U_RK4[Delta_t[-1]][3,:]**2) , c='magenta', label=r'$\Delta t =$ '+str(Delta_t[-1]))
ax.legend(loc=0, fancybox=False, edgecolor="black", ncol = 1, fontsize=16)

ax = axes[2]
ax.set_xlim(0,t)
ax.set_title('Conservation of kinetic energy with Crank-Nicolson scheme',fontsize=20)
ax.grid()
ax.set_xlabel(r'$t$',fontsize=20)
ax.set_ylabel(r'$(\dot{x}^2 + \dot{y}^2)^{\frac{1}{2}}$',fontsize=20)
ax.plot( np.linspace(0,t,len(U_CrankNic[Delta_t[0]][0,:])), np.sqrt(U_CrankNic[Delta_t[0]][2,:]**2 + U_CrankNic[Delta_t[0]][3,:]**2) , c='green', label=r'$\Delta t =$ '+str(Delta_t[0]))
ax.plot( np.linspace(0,t,len(U_CrankNic[Delta_t[1]][0,:])), np.sqrt(U_CrankNic[Delta_t[1]][2,:]**2 + U_CrankNic[Delta_t[1]][3,:]**2) , c='purple', label=r'$\Delta t =$ '+str(Delta_t[1]))
ax.plot( np.linspace(0,t,len(U_CrankNic[Delta_t[-1]][0,:])), np.sqrt(U_CrankNic[Delta_t[-1]][2,:]**2 + U_CrankNic[Delta_t[-1]][3,:]**2) , c='magenta', label=r'$\Delta t =$ '+str(Delta_t[-1]))
ax.legend(loc=0, fancybox=False, edgecolor="black", ncol = 1, fontsize=16)

plt.show()
