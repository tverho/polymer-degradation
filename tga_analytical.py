# Analytical prediction of the TGA curve

import numpy as np
import scipy.special as sc
from math import *
import sys

def volatile_fraction(mu, k):
    return 1 - (1+k/mu)*((mu-1)/mu)**k

def volatile_fraction_xl(mu, k, f_xl):
    q_xl = 1 - f_xl
    q_t = q_xl*(1-1/mu)
    unxl_total = q_xl/(mu+q_xl-mu*q_xl)**2
    unxl_volatile = unxl_total * (1 - (1+k-k*q_t)*q_t**k)
    single_xl_frac = (1-q_xl)*(q_t+1)/(mu**2*(q_t-1)**3)
    dimer_frac = np.zeros_like(q_xl)
    dimer_frac[f_xl != 0] = single_xl_frac**2/(1-unxl_total)
    dimer_frac[f_xl == 0] = 0
    A = k**4 + 4*k**3 + 5*k**2 + 2*k
    B = -4*k**4 - 12*k**3 - 2*k**2 + 18*k + 12
    C = 6*k**4 + 12*k**3 - 12*k**2 - 18*k + 12
    D = -4*k**4 - 4*k**3 + 10*k**2 - 2*k
    E = k**4 - k**2
    xl_volatile = dimer_frac*(1 - (A*q_t**k + B*q_t**(k+1) + C*q_t**(k+2) + D*q_t**(k+3) + E*q_t**(k+4))/(12*q_t*(q_t+1)))
    return unxl_volatile + xl_volatile


N = 1e5
n_chains0 = 400
n_xl0 = n_chains0*1.2
n_bonds_0 = N - n_chains0 + n_xl0

A = 1.0e14
E = 248500
R = 8.314
beta = 10./60. # heating rate K/s

w0 = 67.328
w1 = 1191.8
w2 = 0.90918
w3 = 20.941

T0 = 273.15 + 100.0

T = np.linspace(T0, 273.15+500, 10000)
t = (T-T0)/beta
b = E/(R*beta)
c = T0/beta
decay = np.exp(-A*(b*sc.expi(-b/(c+t))+(c+t)*np.exp(-b/(c+t))))
n_bonds = n_bonds_0 * decay
n_xl = n_xl0 * decay
n_chains = N - n_bonds + n_xl
mu = N / n_chains
k = w3/((w1-w0)/(T-w0)-1)**(1/w2)
m = 1 - volatile_fraction_xl(mu, k, 2*n_xl/N)
tga = m/m[0]

np.savetxt(sys.stdout, np.column_stack([t, T-273.15, 100*tga]))
