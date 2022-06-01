# Fitting experimental TGA data with analytical curves

import numpy as np
import pandas as pd
from matplotlib import pyplot
import scipy.special as sc
from scipy import constants, optimize
from math import *
from collections import namedtuple
import sys
import tomli

def volatile_fraction(mu, k):
    return 1 - (1+k/mu)*((mu-1)/mu)**k

def volatile_fraction_xl(mu, k, f_xl):
    q_xl = 1 - f_xl
    q_t = q_xl*(1-1/mu)
    unxl_total = q_xl/(mu+q_xl-mu*q_xl)**2
    unxl_volatile = unxl_total * (1 - (1+k-k*q_t)*q_t**k)
    single_xl_frac = (1-q_xl)*(q_t+1)/(mu**2*(q_t-1)**3)
    dimer_frac = np.divide(single_xl_frac**2, 1-unxl_total, where=unxl_total!=1, out=np.zeros_like(q_xl))
    #dimer_frac = single_xl_frac**2/(1-unxl_total)
    A = k**4 + 4*k**3 + 5*k**2 + 2*k
    B = -4*k**4 - 12*k**3 - 2*k**2 + 18*k + 12
    C = 6*k**4 + 12*k**3 - 12*k**2 - 18*k + 12
    D = -4*k**4 - 4*k**3 + 10*k**2 - 2*k
    E = k**4 - k**2
    numer = (A*q_t**k + B*q_t**(k+1) + C*q_t**(k+2) + D*q_t**(k+3) + E*q_t**(k+4))
    denom = (12*q_t*(q_t+1))
    xl_volatile = dimer_frac*(1 - np.divide(numer, denom, where=denom!=0, out=np.zeros_like(numer)))
    return unxl_volatile + xl_volatile


def tga_curve(Mw, xl_density, A, E, T_arr, t_arr, pre_evap_T=None):
    """ Mw in g/mol, xl_density in crosslinks per monomer unit, T in Kelvin, t in seconds""" 
    mean_length = Mw/2 / 14
    N = 1.0   # This value doesn't matter
    n_chains0 = N / mean_length
    n_xl0 = xl_density * N
    n_bonds_0 = N - n_chains0 + n_xl0

    #A = 1.0e14
    #E = 248500
    R = 8.314

    w0 = 67.328
    w1 = 1191.8
    w2 = 0.90918
    w3 = 20.941

    T, t = T_arr, t_arr
    #T = np.linspace(T_start, T_end, 10000)
    T0 = T[0]
    
    beta = np.gradient(T, t)
    b = E/(R*beta)
    c = T0/beta
    decay = np.exp(-A*(b*sc.expi(-b/(c+t))+(c+t)*np.exp(-b/(c+t))))
    n_bonds = n_bonds_0 * decay
    n_xl = n_xl0 * decay
    n_chains = N - n_bonds + n_xl
    mu = N / n_chains
    k = w3/((w1-w0)/(T-w0)-1)**(1/w2)
    m = 1 - volatile_fraction_xl(mu, k, 2*n_xl/N)
    if pre_evap_T is not None:
        k0 = w3/((w1-w0)/(pre_evap_T-w0)-1)**(1/w2)
        m0 = 1 - volatile_fraction_xl(mu[0], k0, 2*n_xl[0]/N)
    else:
        m0 = m[0]
    tga = m/m0
    tga[tga>1] = 1
    return tga


def fit_tga_test(tga, T, t, Mw_guess, xl_density_guess, A_guess, E_guess):
    
    def fitfunc(pars):
        Mw, xl_density, A, E = pars
        return np.sum((tga_curve(Mw, xl_density, A, E, T, t) - tga)**2)

    X0 = [Mw_guess, xl_density_guess, A_guess, E_guess]
    #bounds = ([1, 0, 1e10, 100e3], [np.inf, 1, np.inf, np.inf])
    bounds = [(100, 1e5), (0, 0.1), (1e10, 1e20), (100e3, 400e3)]
    res = optimize.differential_evolution(fitfunc, x0=X0, bounds=bounds, tol=0.001)
    Mw, xl_density, A, E = res.x
    return Mw, xl_density, A, E, res.fun


# Function to unpack values
vars = ('Mw', 'xl_density', 'A', 'E')
def unpack_var_values(pars, n_datasets, config):
    n = n_datasets
    pars = list(pars)
    var_values = dict()
    for var in vars:
        if config[var].type == 'fit_free':
            var_values[var] = np.array(pars[:n])
            pars = pars[n:] 
        elif config[var].type == 'fit_single':
            var_values[var] = np.broadcast_to(pars.pop(0), n)
        else:
            var_values[var] = np.broadcast_to(config[var].value, n)
    return var_values

def fitfunc(X, n_datasets, config):
    n = n_datasets
    values = unpack_var_values(X, n, config)
    Mw, xl_density, A, E = values['Mw'], values['xl_density'], values['A'], values['E']
    pre_evap_T = config.get('pre_evap_T', None)
    residuals = []
    for i in range(n):
        model_curve = tga_curve(Mw[i], xl_density[i], A[i], E[i], T_arrs[i], t_arrs[i], pre_evap_T=pre_evap_T)
        residuals.append(model_curve - tga_arrs[i])
    #return np.concatenate(residuals)
    return np.sum(np.concatenate(residuals)**2)

def fit_tga(tga_arrs, T_arrs, t_arrs, config):
    n = len(tga_arrs)
        
    # Pack values and bounds to a vector
    X0 = []
    bounds = []
    for var in vars:
        if config[var].type == 'fit_free':
            X0 += np.broadcast_to(config[var].guess, n).tolist()
            bounds += [config[var].bounds]*n
        elif config[var].type == 'fit_single':
            X0.append(config[var].guess)
            bounds.append(config[var].bounds)
    
    # Perform minimization
    # lsq_bounds = ([b[0] for b in bounds], [b[1] for b in bounds])
    # res = optimize.least_squares(fitfunc, X0, bounds=lsq_bounds)
    #funcval = res.cost
    res = optimize.differential_evolution(fitfunc, bounds=bounds, args=(n, config), 
        workers=-1, init='sobol', x0=X0, updating='deferred', tol=0.001, popsize=30, maxiter=10000)
    #res = optimize.dual_annealing(fitfunc, bounds=bounds, initial_temp=500)
    funcval = res.fun
    
    # Unpack results
    result = unpack_var_values(res.x, n, config)
    
    print(f'{res.nit} iterations')
    return result, funcval
    

def load_datafile(filename, start_T=None, default_beta=10/60):
    data = np.loadtxt(filename)
    if data.shape[1] == 2:
        T, tga = data.T
        t = (T-T[0])/default_beta
    elif data.shape[1] == 3:
        t, T, tga = data.T
    else:
        raise Exception('Unexpected number of data columns')
    
    if T[0] < 300:
        T += 273.15
    if np.max(tga) > 1:
        tga /= 100
        
    if start_T is not None:
        idx = np.flatnonzero(T>start_T)[0]
        t = t[idx:] - t[idx]
        T = T[idx:]
        tga = tga[idx:]
        
    return t, T, tga

def ensure_kelvin(val):
    if isinstance(val, str):
        if val[-1].upper() == 'C':
            return 273.15 + float(val[:-1])
        elif val[-1].upper() == 'K':
            return float(val[:-1])
    return float(val)

with open(sys.argv[1], 'rb') as f:
    config = tomli.load(f)

files = config['data']['paths']

T_arrs, tga_arrs, t_arrs = [], [], []
for f in files:
    t, T, tga = load_datafile(f)
    T_arrs.append(T)
    tga_arrs.append(tga)
    t_arrs.append(t)

# A nicer version of the config dict
conf = dict()
entry = namedtuple('entry', ['type', 'guess', 'value', 'bounds'], defaults=[None, None, (None, None)])
conf['Mw'] = entry(**config['Mw'])
conf['xl_density'] = entry(**config['xl_density'])
conf['A'] = entry(**config['A'])
conf['E'] = entry(**config['E'])

if config['settings'].get('pre_evap_T') not in [None, '']:
    conf['pre_evap_T'] = ensure_kelvin(config['settings']['pre_evap_T'])

# Perform fit
result, funcval = fit_tga(tga_arrs, T_arrs, t_arrs, conf)

# Scissions
bonds_density = 1 - 1/(result['Mw']/2 / 14)
scissed = 1 - bonds_density/bonds_density[0]
result['scission_frac'] = scissed
    
# r(450 C)
r450 = result['A']*np.exp(-result['E']/(constants.R*(500+273.15)))
result['r(450 C)'] = r450

# Print results
df = pd.DataFrame(result)
print(f'Function value at minimum: {funcval:.4g}')
print(df)

    
# Plot
labels = config['data']['labels']
for T, tga, label in zip(T_arrs, tga_arrs, labels):
    pyplot.plot(T-273.15, tga, label=label)

pyplot.gca().set_prop_cycle(None)
for i in range(len(tga_arrs)):
    curve = tga_curve(result['Mw'][i], result['xl_density'][i], result['A'][i], result['E'][i], T, t, pre_evap_T=conf['pre_evap_T'])
    pyplot.plot(T-273.15, curve, '--')
pyplot.xlim(None, 550)
pyplot.legend()
pyplot.show()
#pyplot.savefig(sys.argv[1].replace('.toml', '.pdf'))

