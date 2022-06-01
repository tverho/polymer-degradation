# Analytic calculation of gel fraction with aging equations derived from ODEs

from math import *
import numpy as np
from matplotlib import pyplot
import sys

N = 1e6
Mw = 5600
n_chains = N / (Mw/14/2)
n_xl = n_chains*0.6
rho_b0 = 1 - n_chains/N + n_xl/N

t = np.logspace(0, 6.3, 1000)

g_ax = pyplot.figure().gca()
loops_ax = pyplot.figure().gca()
#xlinks_ax = pyplot.figure().gca()

for p_xl in (0.2,): #(0, 0.18, 0.2, 0.25, 0.33, 0.5):
    r_xl = p_xl*1/N
    r_sc = (1-p_xl)*1/N
    r_b = r_xl-r_sc
    
    if r_b != 0:
        rho_c = (n_chains/N - 1)*(t*r_b/rho_b0 + 1)**(-r_sc/r_b) + 1
        rho_xl = rho_c - 1 + rho_b0 + r_b*t
    else:
        rho_xl = r_xl/r_sc + (n_xl/N -r_xl/r_sc)*np.exp(-r_sc*t)
        rho_c = rho_xl + 1 - rho_b0

    kf = 2*rho_xl/rho_c
    _kf = np.maximum(kf, 0.5)
    g = (_kf-2 + np.sqrt(_kf**2 + 4*_kf))/(2*_kf)
    
    chains_in_gel = rho_c * (1 - np.sqrt(1-g))
    xlinks_in_gel = rho_xl * (1 - (1-g)**2)
    rho_loops = xlinks_in_gel - chains_in_gel
    
    rho_bonds = 1 - rho_c + rho_xl
    
    g_ax.semilogx(t/N*(1-p_xl), g, label=f'p(xl)={p_xl}')
    loops_ax.loglog(t/N*(1-p_xl), rho_loops, label=f'p(xl)={p_xl}')
    #xlinks_ax.plot(t, rho_xl, label=f'p(xl)={p_xl}')
    
    np.savetxt(sys.stdout, np.column_stack([t, g, rho_loops, rho_bonds]))
    

g_ax.set_xlabel('Aging reactions')
g_ax.set_ylabel('Gel content')
loops_ax.set_xlabel('Aging reactions')
loops_ax.set_ylabel(r'$n_{loops}/n_{monomers}$')
loops_ax.set_ylim(1e-6, 1e-2)
#xlinks_ax.set_xlabel('Aging reactions')
#xlinks_ax.set_ylabel('Cross-link density')
g_ax.legend()
loops_ax.legend()
#xlinks_ax.legend()
g_ax.figure.savefig('aging_gel_content.png')
loops_ax.figure.savefig('aging_loops.png')
pyplot.show()
