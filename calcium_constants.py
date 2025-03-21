import numpy as np

# Physics constants
hplanck = 6.62607015e-34
hbar = (hplanck/2/np.pi)
kb   = 1.380649 * 1e-23
c_light = 299792458
C_e  = 1.602176634e-19
eps0 = 8.85418782e-12
Coul_factor = 1/(4*np.pi*eps0) # * C_e**2/m_Ca

m_Ca  = 40.078*1.66053906660e-27
Gamma_SP = 21.57e6*2*np.pi
f_397_Wan = 755.222765896*1e12