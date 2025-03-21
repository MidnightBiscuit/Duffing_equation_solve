import numpy as np
import math

from scipy.integrate import solve_ivp
from scipy.interpolate import approximate_taylor_polynomial

import os
import sys
sys.path.append("../../Functions")
from calcium_constants import *

import time
from datetime import datetime

class DDDAO_3:
    '''
        Class to represent a 1D Doubly-Driven and Damped Anharmonic Oscillator (DDDAO).
        Uses the variable time-step method `solve_ivp` from `scipy` module to solve the ODE.
        The drives are the electric tickle and the NW as modelled by M. Weegen in his thesis.
        The solution can be computed for a range of parameters swept over time.
        The solution is first initialised, then alternates between a steady-state and a sweep.
        For each steady-state part, only the last part of the dynamics is saved to limit the memory usage.

        Based on
        N. Akerman, S. Kotler, Y. Glickman, Y. Dallal, A. Keselman, and R. Ozeri.
        Single-ion nonlinear mechanical oscillator, PRA 82 (2010).
        https://journals.aps.org/pra/abstract/10.1103/PhysRevA.82.061402.
    '''

    def __init__(self,
                V_tkl = 0.025, V_nw = 1, V_piezo = 1,
                omega_z = 422500*2*np.pi, omega_drive = 422500*2*np.pi,
                phi_tkl = 0, phi_nw = 0,                
                B = -4.84e19,
                cooling_rate = None, nl_damping_rate = None,
                P397 = 75e-6, w397 = 120e-6, lam_397 = 396.959120e-9,
                i_init = None, i_drive = None, i_freq_sweep = None, i_smooth_sweep = None,
                solve_ivp_method = 'RK45',
                sweep_length = 21,
                add_string=100):
        '''
        Initialise the physical parameters of the trap coupling with nw, tickle and laser.
        Also initialise the parameters of the numerical simulation itself.
        Parameters
        ----------
        V_tkl : array-like or float
            Voltage applied to the tickle electrode.
        V_nw : array-like or float
            Voltage applied to the nanowire.
        V_piezo : array-like or float
            Voltage applied to the piezo element driving the oscillation of the nanowire.
            See Moritz thesis for more details about the modelling.
        omega_z : array-like or float
            Angular frequency of the harmonic potential trapping the particle.
        omega_drive : array-like or float
            Angular frequency of the drives.
        phi_tkl : array-like or float
            Phase of the tickle drive.
        phi_nw : array-like or float
            Phase of the nanowire drive.
        B : array-like or float
            Cubic coefficient of the trapping acceleration such that
            $a_{trap} = a_3 = \omega_z^2z + Bz^3$.
        cooling_rate : array-like or float or None
            Linear coefficient of friction that represents laser-cooling.
            If None, this value is computed using a first-order Taylor expansion of the scattering rate.
        nl_damping_rate : array-like or float or None
            Non-linear coefficient of friction that represents laser-cooling such that
            $a_{cool} + a_{a_nl_damping} = a_4 + a_5 = cooling_rate\dot{x} + nl_damping_rate\dot{x}^3$.
            If None, this value is computed using a third-order Taylor expansion of the scattering rate.
        P397 : array-like or float
            Power of the laser.
        w397 : array-like or float
            Beam radius.
        lam_397 : array-like or float
            Laser wavelength.
        i_init : float or None
            Duration of initialisation in the unit of the number of secular periods ($2\pi/omega_z$).
            If None it is set to 2500.
        i_drive : float or None
            ONLY SET TO NONE PLZ.
        i_freq_sweep : float or None
            Duration of steady-state solution in the unit of the number of secular periods ($2\pi/omega_z$).
            If None it is set to 5000.
        i_smooth_sweep : float or None
            Duration of parameter sweep in the unit of the number of secular periods ($2\pi/omega_z$).
            If None it is set to 10.
        solve_ivp_method : string or OdeSolver
            Method used by `scipy.solve_ivp` to solve the differential equation.
            'RK45' is default, but 'LSODA' is a good candidate.
        sweep_length : int
            Number of steps in the parameter sweep.
        add_string : int
            String to be added at the end of the data file names. This value is incremented for each sweep step.
        '''

        # Retrives the variables passed as arguments of the class.
        self.arguments = locals()
        self.add_string = self.arguments['add_string']

        print('=== Initialise simulation ===')

        # self.omega_drive0 = omega_drive[0] if type(omega_drive) is list or type(omega_drive) is np.ndarray else omega_drive
        # self.omega_drive_aux = omega_drive

        # NW coupling parametrers
        self.d_offset = 33e-6
        # tickle parameters
        self.d_tkl = 3e-3 /2
        self.delta_grad_V0 = 3.2132701249798 # V/m /Vtkl  axial gradient difference for 1 Vtkl amplitude
        self.factor_tkl = 0.9025798077916855 # to make tkl and nw equivalent given XP conditions

        # heat parameters
        self.kheat = 0 # legacy

        # update the variables so it is ready for the sweep
        self.update_arguments()
        self.update_parameters(0,0,1)

        # numerical parameters
        self.n_dt = 1000 # Resolution of solution interpolation for the data output (number of time-steps for one omega_z period)
        self.dt = 2*math.pi/(self.n_dt*self.omega_z) # Interpolation time-step duration
        
        self.i_init       = i_init       if i_init is not None else 2500
        self.i_drive      = i_drive      if i_drive is not None else 0
        self.i_smooth_sweep = i_smooth_sweep if i_smooth_sweep is not None else 5000
        self.i_freq_sweep   = i_freq_sweep if i_freq_sweep is not None else 10
        # Have simulation last i_xxx secular periods
        self.n_init  = self.n_dt*self.i_init # Total number of time-steps for init
        self.n_drive = self.n_dt*self.i_drive # Total number of time-steps for drive
        self.n_smooth_sweep = self.n_dt*self.i_smooth_sweep # Total number of time-steps for one smooth_sweep step
        self.n_freq_sweep   = self.n_dt*self.i_freq_sweep # Total number of time-steps for one freq_sweep step

        # Total number of time-steps for the whole simulation
        self.n_total = self.n_init + self.n_drive + (self.n_smooth_sweep + self.n_freq_sweep)*sweep_length - self.n_smooth_sweep

        self.time = [] # time array
        self.time_switch = [] # time array when simu is switched from one to another kind (init, steady-state, sweep)
        # arrays to save drive frequency (for self-checking in analysis)
        self.omega_drive_save = []
        self.omega_tkl_save = []
        self.omega_nw_save = []

        # Initialise r and v
        self.rva_init = [2e-6,0] # [pos,vel]

        self.k_sweep = 0 # current sweep step
        self.solve_ivp_method = solve_ivp_method # solve method

    def scattering_rate(self,v,gamma,s,k,delta):
        '''
        The scattering rate as defined by the electric dipole, RWA approximation,
        for a two-levels system.

        Parameters :
            v : scalar
                The velocity of the ion.
            gamma : scalar
                The linewidth of the transition.
            s : scalar
                The saturation parameter such as s = I/Isat.
            k : scalar
                The wavevector of the cooling laser such as k = 2pi/wavelength.
            delta : scalar
                The detuning of the cooling laser such as delta = omega_l - omega_0.
        Returns :
            rho_ee : scalar
                The excitation probability.
        '''

        return 0.5*gamma*s/(1+s+4/gamma**2*(delta-k*v)**2)

    def laser_parameters(self):
        '''
        Laser parameters. Doppler cooling with 397 only.
        '''
        self.k397 = 2*math.pi/self.lam_397 # wavenumber
        self.f_397 = c_light/self.lam_397  # laser frequency
        self.detuning = (self.f_397 - f_397_Wan)*2*math.pi # detuning
        self.intensity = 4*self.P397/(math.pi*self.w397**2)
        # self.satI = 2*hbar*math.pi**2*c_light/(3*self.lam_397**3)*Gamma_SP
        self.satI = hbar*(2*math.pi*self.f_397)**3/(12*math.pi*c_light**2)*Gamma_SP # sat intensity
        self.sat0 = self.intensity/self.satI

        # self.Rs_scipy = approximate_taylor_polynomial(lambda v: self.scattering_rate(v,Gamma_SP,self.sat0,self.k397,self.detuning), 0, 3, 0.1)
    
    def friction_parameters(self):
        '''
        Constants of the anharmonicities and friction.
        '''
        # self.beta = -4*self.detuning/Gamma_SP * self.sat0 / (1+self.sat0+4*(self.detuning/Gamma_SP)**2)**2 # no unit
        # self.mu_laser = self.beta * hbar*self.k397**2/m_Ca
        self.beta  = -self.K_Rs_1 # no unit
        self.mu_laser = self.beta * hbar*self.k397/m_Ca /2 # linear damping
        self.gamma_laser = np.abs(self.K_Rs_3 * hbar*self.k397/m_Ca) # non-linear damping

    def taylor_coefficients(self):
        '''
        Coefficients of Taylor expansion of the scattering rate
        up to third-order included.
        '''
        self.Rs_0 = 1/(1+self.sat0+4*(self.detuning/Gamma_SP)**2)
        self.K_Rs_0 = self.Rs_0                                                   * Gamma_SP/2*self.sat0
        self.K_Rs_1 = + 8 /Gamma_SP**2*self.detuning * self.Rs_0**2 * self.k397   * Gamma_SP/2*self.sat0
        self.K_Rs_2 = - 4/Gamma_SP**2 * (self.sat0-12/Gamma_SP**2*self.detuning**2+1) * self.Rs_0**3 * self.k397**2                      * Gamma_SP/2*self.sat0
        self.K_Rs_3 = - 4*16/Gamma_SP**4 * self.detuning * (self.sat0+1 - 4/Gamma_SP**2*self.detuning**2) * self.Rs_0**4 * self.k397**3  * Gamma_SP/2*self.sat0

        # print(K_Rs_0, K_Rs_1, K_Rs_2, K_Rs_3)
        # print(hbar*k/m_Ca * K_Rs_1/2/2/np.pi)

    def update_arguments(self):
        '''
        Update the class arguments to be lists of the same length sweep_length.
        If arguments are non array-like, they are transformed into a list of length sweep_length.
        '''
        
        self.sweep_length = self.arguments['sweep_length']
        # print(self.sweep_length)

        for i,j in enumerate(self.arguments):
            # print(f'Eval. {self.arguments[j]} now')
            if isinstance(self.arguments[j],(list,tuple,np.ndarray)) == True:
                # print('list')
                # print(len(self.arguments[j]))
                if len(self.arguments[j]) != int(self.sweep_length):
                    print(f'{i:02d}th arg. = {j} = has wrong length {len(self.arguments[j])} rather than {self.sweep_length} !')
            elif isinstance(self.arguments[j],(str)) == True:
                # print('string')
                pass
            elif isinstance(self.arguments[j],(int,float,complex)) == True:
                # print('scalar')
                self.arguments[j] = [self.arguments[j]]*self.sweep_length

        # prints all the arguments transformed
        for i,keys in enumerate(self.arguments):
            print(f'  {i:02d}  {keys:17s} {self.arguments[keys]}')

    def update_parameters(self,i_time,sweep_step,lftria,prints=0):
        '''
        Update the physical parameters of the experiment. The variables used by
        the solution algorithm are updated to their right value. During a sweep
        this function adds the proper value. The sweep is based on a tanh function.

        Parameters :
            i_time : float
                float between 0 and 1. It starts at 0 at the beginning of a sweep
                and is 1 at the very end of the sweep. Evolves linearly with the
                progression of the sweep.
            sweep_state : int
                The index indicating which step it is in the sweep. Maximum value is
                sweep_length.
            lftria : int
                Activating the sweep depending its state : 0 for no sweep, 1 for sweep.
                It should be 0 during a steady-state experiment, 1 during a sweep.
        '''

        N_H = 4 # order in the tanh for sweep function

        # the tanh based sweep function
        '''
        Jofre Pedregosa-Gutierrez, Caroline Champenois, Marius Romuald Kamsap, Martina Knoop,
        Ion transport in macroscopic RF linear traps,
        International Journal of Mass Spectrometry,
        Volumes 381–382,
        2015,
        https://doi.org/10.1016/j.ijms.2015.03.008.
        '''
        tanh_curve = lftria * (math.tanh(2*N_H*i_time-N_H)/math.tanh(N_H) + 1)

        self.V_tkl   = self.arguments['V_tkl'][sweep_step] + np.diff(self.arguments['V_tkl'],append=0)[sweep_step]/2 * tanh_curve
        self.V_nw    = self.arguments['V_nw'][sweep_step] + np.diff(self.arguments['V_nw'],append=0)[sweep_step]/2 * tanh_curve
        self.V_piezo = self.arguments['V_piezo'][sweep_step] + np.diff(self.arguments['V_piezo'],append=0)[sweep_step]/2 * tanh_curve

        self.omega_z = self.arguments['omega_z'][sweep_step] + np.diff(self.arguments['omega_z'],append=0)[sweep_step]/2 * tanh_curve
        self.omega_drive = self.arguments['omega_drive'][sweep_step] + np.diff(self.arguments['omega_drive'],append=0)[sweep_step]/2 * tanh_curve
        self.omega_tkl = self.omega_drive
        self.omega_nw  = self.omega_drive

        self.phi_tkl = self.arguments['phi_tkl'][sweep_step] + np.diff(self.arguments['phi_tkl'],append=0)[sweep_step]/2 * tanh_curve
        self.phi_nw  = self.arguments['phi_nw'][sweep_step] + np.diff(self.arguments['phi_nw'],append=0)[sweep_step]/2 * tanh_curve

        self.B       = self.arguments['B'][sweep_step] + np.diff(self.arguments['B'],append=0)[sweep_step]/2 * tanh_curve

        if self.arguments['cooling_rate'] == None:
            self.P397    = self.arguments['P397'][sweep_step] + np.diff(self.arguments['P397'],append=0)[sweep_step]/2 * tanh_curve
            self.w397    = self.arguments['w397'][sweep_step] + np.diff(self.arguments['w397'],append=0)[sweep_step]/2 * tanh_curve
            self.lam_397 = self.arguments['lam_397'][sweep_step] + np.diff(self.arguments['lam_397'],append=0)[sweep_step]/2 * tanh_curve

            self.laser_parameters() # physical quantities
            self.taylor_coefficients() # Taylor expansion to third order of scattering rate
            self.friction_parameters() # set fritcion parameters in Duffing equation

            self.cooling_rate    = 2*self.mu_laser # * 30 # cooling_rate or self.mu_laser * 30
            self.nl_damping_rate = self.gamma_laser
        else:
            self.cooling_rate    = self.arguments['cooling_rate'][sweep_step] + np.diff(self.arguments['cooling_rate'],append=0)[sweep_step]/2 * tanh_curve
            self.nl_damping_rate = self.arguments['nl_damping_rate'][sweep_step] + np.diff(self.arguments['nl_damping_rate'],append=0)[sweep_step]/2 * tanh_curve       

        self.eps  = Coul_factor*C_e*1.844*1e-15*self.V_nw
        self.d_nw = 225e-6 + self.d_offset
        self.z_nw = 100e-6 * self.d_nw / (self.d_nw-self.d_offset)        

        self.omega_z_2 = self.omega_z **2
        self.A_nw    = 184.41690653263915e-9*self.V_piezo
        self.A1      = ( - self.eps/(self.d_nw**3) * (1 - 3*self.z_nw**2/self.d_nw**2) ) * self.A_nw
        self.A2      = - C_e * self.V_tkl * self.delta_grad_V0 # *self.factor_tkl*self.V_nw*self.V_piezo # 0.0028

        if prints == 1 or prints == True:
            print('  --- Potential parameters ---')
            print(f'  omega_z = {self.omega_z:.3e}')
            print(f'  omega_drive0  = {self.omega_drive:.3e}')
            print(f'  Pot : B = {self.B:.5e}')

            print('  --- Coupling parameters ---')
            print(f'  V_tkl = {self.V_tkl:.3e}')
            print(f'  V_nw  = {self.V_nw:.3e}')
            print(f'  V_piezo = {self.V_piezo:.3e}')
            print(f'  phi_tkl = {self.phi_tkl:.3e}')
            print(f'  phi_nw  = {self.phi_nw:.3e}')
            print(f'  NW  : A1/m = {self.A1/m_Ca:.3e}')
            print(f'  Tkl : A2/m = {self.A2/m_Ca:.3e}')

            print('  --- Laser parameters ---')
            print(f'  Saturation = {self.sat0:.5e}')
            print(f'  Detuning   = {self.detuning/Gamma_SP:.5e} Gamma')

            print(f'  Beta       = {self.beta:.5e}') # no unit
            print(f'  Gamma      = {self.mu_laser:.5e}')
            
            print(f'  cooling_rate    = {self.cooling_rate:.5e}')
            print(f'  nl_damping_rate = {self.nl_damping_rate:.5e}')
            print('  Coefficients of the Taylor expansion for laser cooling')
            print('  Orders         0       1           2          3')
            hkm = hbar*self.k397/m_Ca
            print(f'          {hkm * self.K_Rs_0:.3e}, {hkm * self.K_Rs_1:.3e}, {hkm * self.K_Rs_2:.3e}, {hkm * self.K_Rs_3:.3e}')
        
    # Expressions for the various forces
    def a_trap(self, el_tiempo, la_posicion):
        return self.omega_z_2*(la_posicion) + self.B*la_posicion**3

    def a_cool(self, la_velocidad):
        return self.cooling_rate*la_velocidad

    def a_tickle(self, el_tiempo):
        return self.A2/m_Ca*math.cos(self.omega_tkl * el_tiempo + self.phi_tkl)

    def a_nw(self, el_tiempo):
        return self.A1/m_Ca*math.cos(self.omega_nw * el_tiempo + self.phi_nw)

    def a_nl_damping(self, la_velocidad):
        '''
        Non-slinear damping force, choosen as cubic velocity term.
        '''
        return self.nl_damping_rate*la_velocidad**3

    # Derivatives
    def derivs_init(self,el_tiempo,la_posicion,la_velocidad):
        self.dydx_temp  = la_velocidad

        a_3 = self.a_trap(el_tiempo, la_posicion)
        a_4 = self.a_cool(la_velocidad)
        a_5 = self.a_nl_damping(la_velocidad)
        self.dydx2_temp = - a_3 - a_4 - a_5

    def derivs(self,t,rva):
        '''
        Compute the derivative for the RK4 algorithm.

        Parameters :
            t : array
                The time.
            rva : array
                The variable of the dynamics (position, velocity).
        Returns :
            rva : array
                The updated variable of the dynamics.
        '''
        
        i_time = (t - self.time[0])/(self.time[-1] - self.time[0])
        self.update_parameters(i_time,self.k_sweep,self.lftria,prints=math.floor(1-i_time))

        self.omega_drive_save.append(self.omega_drive)
        self.omega_tkl_save.append(self.omega_tkl)
        self.omega_nw_save.append(self.omega_nw)

        self.dydx_temp  = rva[1]

        a_1 = self.a_tickle(t)
        a_2 = self.a_nw(t)
        a_3 = self.a_trap(t, rva[0])
        a_4 = self.a_cool(rva[1])
        a_5 = self.a_nl_damping(rva[1])
        self.dydx2_temp = a_1 + a_2 - a_3 - a_4 - a_5
        #           chpeed       acchelerachion
        return [self.dydx_temp, self.dydx2_temp]

    def run_ivp_solve(self,time,rva_init):
        '''
        Solves the ODE with a variable time step. Uses `solve_ivp` from `scipy` module.

        Parameters :
            time : array
                The time for which solution should be saved.
            rva_init : array
                The variable of dynamics to start with (position, velocity).
        Returns :
            sol : OdeSolution or None
                Found solution as OdeSolution instance; None if dense_output was set to False.
                See help(solve_ivp) for more informations.
        '''

        print('> Run ivp_solve')
        sol = solve_ivp(self.derivs,
                    t_span = [time[0], time[-1]], # initial time and final time
                    y0 = self.rva_init, # initial values
                    method=self.solve_ivp_method,
                    t_eval=time, # time at which output should be evaluated
                    rtol = 1e-6) # relative error tolerance
        return sol

    # Run init with all forces
    def run_init(self):
        '''
        Runs initialisation step.
        '''
        print('')
        print('=== Run init ===')
        self.lftria = 0 # 0 no sweep

        # set time
        self.time = np.linspace(0, self.n_init*self.dt, self.n_init)
        # print(len(self.time))
        # print(f'  dt = {np.diff(self.time)[2]}')
        self.time_switch.append([0,0,self.time[0],self.time[-1]])

        # solve the equation for init
        start_time = time.time()
        self.sol = self.run_ivp_solve(self.time,self.rva_init)
        end_time = time.time()

        # print(np.shape(self.sol.t))
        # print(np.shape(self.time))
        # print(np.shape(self.omega_drive_save))

        self.i_sweep = 0
        
        self.time_monitor(start_time,end_time)

        self.create_filenames()
        self.save_data_end((int(self.n_init*0.95),self.n_init))
        # self.save_data()
        self.save_metadata()

        self.reset_rva()

    # Run drive
    def run_drive(self):
        '''
        Runs the full model with trap and laser cooling, tickle and NW.
        DO NOT USE AS IS. Needs to be updated to use solve_ivp.
        '''
        print('')
        print('> DRIVE')
        print('')
        # print(self.time[self.n_init])
        for i,j in enumerate(self.time[self.n_init:self.n_total]):
            self.update_rk4(self.derivs)
            self.update_rva()

        # self.save_data()

    def run_freq_sweep(self):
        '''
        Runs the full model with trap and laser cooling, tickle and NW.
        Alternate between two kind of simulations :
            steady-state : parameters are kept constant and equation is solved.
            sweep : parameters are swept while equation is solved.
        '''

        print('')
        print('  Will solve equation with')
        print(f'  {self.sweep_length:03d} steps')
        print('')

        for self.k_sweep in range(self.sweep_length):
            print('=== Parameter set ===')
            self.lftria = 0
            print(f'  step number {self.k_sweep:02d}')
            if math.fabs(self.omega_tkl - l) > 0.01*2*math.pi:
                print(f'/!| Frequency missmatch !! by {math.fabs(self.omega_tkl - l):.3f} s-1')

            self.time = np.linspace(self.aux_t, self.aux_t + self.n_freq_sweep*self.dt, self.n_freq_sweep)
            # print(f'  dt = {np.diff(self.time)[0]}')
            self.time_switch.append([self.time[0],self.time[-1]])
            self.sol = self.run_ivp_solve(self.time,self.rva_init)

            # create files and save the data and metadata
            self.i_sweep += 1 # for the index at the end of the save files names
            self.create_filenames()
            try:
                self.save_data_end((-int(self.n_dt * 50), int(self.n_freq_sweep))) # (int(self.n_freq_sweep*0.50), int(self.n_freq_sweep))
            except:
                self.save_data_end((0, int(self.n_freq_sweep)))
            # self.save_data()
            self.save_metadata()

            self.reset_rva()

            # If some duration for the smooth sweep is set
            # then solve the equation for changing omega_drive
            if self.i_smooth_sweep != 0 :
                if self.k_sweep != self.sweep_length-1 :
                    # print('coucou')
                    self.run_smooth_sweep()
                    self.reset_rva()
                else:
                    self.time_switch[self.k_sweep+1].extend([self.time[0],self.time[-1]])

            print('')

    def run_smooth_sweep(self):
        '''
        Solves the equation with variable parameters.
        '''
        print('=== Frequency sweep ===')
        self.lftria = 1 # 1 sweep of parameters
        
        self.time = np.linspace(self.aux_t, self.aux_t + self.n_smooth_sweep*self.dt, self.n_smooth_sweep)
        self.time_switch[self.k_sweep+1].extend([self.time[0],self.time[-1]])
        self.sol = self.run_ivp_solve(self.time,self.rva_init)
    
    def run_end(self):
        '''
        A function executed at the end for additional stuff.
        '''
        self.save_aux()

    def reset_rva(self):
        '''
        Resets the auxiliary variables at the end of each kind.
        A way to keep the last value of dynamic variable to use as
        initial value in the next step.
        '''
        self.aux_t  = self.time[-1]
        aux_r_z     = self.sol.y[0][-1]
        aux_v_z     = self.sol.y[1][-1]
        # aux_a_z     = self.sol.y[2][-1]
        self.rva_init = [aux_r_z, aux_v_z]

    def create_filenames(self):
        '''
        Creates the name for the files where data and metadata is saved.
        '''
        # basis for the filename
        # self.savename = f'ion_Vtkl{self.V_tkl*1e3:06.3f}_VNW{self.V_nw:05.2f}_Vpiezo{self.V_piezo:05.2f}_PhiNW{self.phi_nw*180/math.pi:06.2f}deg{self.add_string}'
        self.savename = f'ion_Vtkl{self.V_tkl*1e3:06.3f}_VNW{self.V_nw:05.2f}_Vpiezo{self.V_piezo:05.2f}_PhiNW{self.phi_nw*180/math.pi:06.2f}deg_{self.add_string+self.i_sweep:04d}'
        self.savedir_date = f'{datetime.today().strftime("%y%m%d")}'

        # directory address to save files
        # self.savedir = f'D:\\Universite_SIMU\\{self.savedir_date}\\'           # WINDOWS
        # self.savedir = f'/Users/adrien/Documents/SIMU/{self.savedir_date}/'    # MACOS
        # self.savedir = f'/swdata/poindron/SIMU/250113/tkl_factor_a_05/'        # Linux SLURM manual
        self.savedir = f'{os.path.dirname(os.path.realpath(__file__))}{os.sep}'  # Linux SLURM auto
        
        print('Save strings names')        
        print(self.savedir)
        print(self.savename)

        # creates the folder
        # do nothing if it already exists thanks to exist_ok=True
        os.makedirs(self.savedir, exist_ok=True)

    def save_data(self):
        '''
        Saves the variables from the dynamics (r_z, v_z, a_z) using `numpy.savez`.
        '''
        print('> SAVE DATA')
        filename = f'{self.savedir}rva_{self.savename}'
        np.savez(filename,
                time=self.time,
                r_z=self.sol.y[0],
                v_z=self.sol.y[1])
                # a_z=self.sol.y[2]
        
    def save_aux(self):
        print('> SAVE AUX')
        filename = f'{self.savedir}aux_{self.savename}'
        np.savez(filename,
                omega_drive_save = self.omega_drive_save[::10],
                omega_tkl_save   = self.omega_tkl_save[::10],
                omega_nw_save    = self.omega_nw_save[::10],
                time_switch = self.time_switch
        )

    def save_data_end(self,limits):
        '''
        Saves the variables from the dynamics (r_z, v_z, a_z) using `numpy.savez`.
        Only keeps the last part according to `limits`.
        '''
        print('> SAVE DATA END')
        filename = f'{self.savedir}rva_{self.savename}'
        # print(limits[1]-limits[0])
        # print(self.time[limits[0]:limits[1]])
        np.savez(filename,
                time = self.time[limits[0]:limits[1]],
                r_z  = self.sol.y[0][limits[0]:limits[1]],
                v_z  = self.sol.y[1][limits[0]:limits[1]])

    def save_metadata(self):
        print('> SAVE METADATA')
        filename = f'{self.savedir}potentials_{self.savename}.txt'
        with open(filename,'w') as f:
            f.write(f'{self.omega_z}\n')
            f.write(f'{self.B}\n')
            f.write(f'{self.d_nw}\n')
            f.write(f'{self.z_nw}\n')
            f.write(f'{self.omega_nw}\n')
            f.write(f'{self.phi_nw}\n')
            f.write(f'{self.V_nw}\n')
            f.write(f'{self.V_piezo}\n')
            f.write(f'{self.V_tkl}\n')
            f.write(f'{self.omega_drive}\n')
            f.write(f'{self.phi_tkl}\n')
            f.write(f'{self.A1/m_Ca}\n')
            f.write(f'{self.A2/m_Ca}\n')
            # f.write(f'{self.kappa_grad}\n')
            # f.write(f'{self.N_H}\n')

        filename = f'{self.savedir}numericals_{self.savename}.txt'
        with open(filename,'w') as f:
            f.write(f'{self.n_dt}\n')
            f.write(f'{self.dt}\n')
            # f.write(f'{self.h}\n')
            f.write(f'{self.n_init:d}\n')
            f.write(f'{self.n_drive:d}\n')
            f.write(f'{self.n_freq_sweep:d}\n')
            f.write(f'{self.n_smooth_sweep:d}\n')
            f.write(f'{self.n_total:d}\n')

        filename = f'{self.savedir}cooling_{self.savename}.txt'
        with open(filename,'w') as f:
            f.write(f'{self.P397}\n')
            f.write(f'{self.w397}\n')
            f.write(f'{self.lam_397}\n')
            f.write(f'{self.kheat}\n')
            f.write(f'{self.sat0}\n')
            f.write(f'{self.detuning}\n')
            f.write(f'{self.beta}\n')
            f.write(f'{self.mu_laser}\n')
            f.write(f'{self.cooling_rate}\n')
            f.write(f'{self.nl_damping_rate}\n')

    def time_monitor(self,start,end):
            print('> Execution time')
            print(f'  For {self.n_init} init steps')
            print(f'{end-start} s')
            print(f'{(end-start)/60} min')

            print('Estimated total wall clock time')
            print(f'  For {self.n_total} total steps')
            print(f'{(end-start)*self.n_total/self.n_init} s')
            print(f'{(end-start)*self.n_total/self.n_init/60} min')
            print(f'{(end-start)*self.n_total/self.n_init/60/60} h')
            print()


def algorithm_print(solve_ivp_method):
    print('')
    if solve_ivp_method == 'LSODA':
        print(r"       Start   numerical   solution    with       ")
        print(r" /$$        /$$$$$$   /$$$$$$  /$$$$$$$   /$$$$$$ ")
        print(r"| $$       /$$__  $$ /$$__  $$| $$__  $$ /$$__  $$")
        print(r"| $$      | $$  \__/| $$  \ $$| $$  \ $$| $$  \ $$")
        print(r"| $$      |  $$$$$$ | $$  | $$| $$  | $$| $$$$$$$$")
        print(r"| $$       \____  $$| $$  | $$| $$  | $$| $$__  $$")
        print(r"| $$       /$$  \ $$| $$  | $$| $$  | $$| $$  | $$")
        print(r"| $$$$$$$$|  $$$$$$/|  $$$$$$/| $$$$$$$/| $$  | $$")
        print(r"|________/ \______/  \______/ |_______/ |__/  |__/")
        print("             a  l  g  o  r  i  t  h  m            ")
    elif solve_ivp_method == 'RK45':
        print(r"    Start   numerical   solution    with    ")
        print(r" /$$$$$$$  /$$   /$$     /$$   /$$ /$$$$$$$ ")
        print(r"| $$__  $$| $$  /$$/    | $$  | $$| $$____/ ")
        print(r"| $$  \ $$| $$ /$$/     | $$  | $$| $$      ")
        print(r"| $$$$$$$/| $$$$$/      | $$$$$$$$| $$$$$$$ ")
        print(r"| $$__  $$| $$  $$      |_____  $$|_____  $$")
        print(r"| $$  \ $$| $$\  $$           | $$ /$$  \ $$")
        print(r"| $$  | $$| $$ \  $$          | $$|  $$$$$$/")
        print(r"|__/  |__/|__/  \__/          |__/ \______/ ")
        print("          a  l  g  o  r  i  t  h  m         ")
    else:
        print(r"          Start         numerical         solution          with          ")
        print(r" /$$   /$$           /$$                                                  ")
        print(r"| $$  | $$          | $$                                                  ")
        print(r"| $$  | $$ /$$$$$$$ | $$   /$$ /$$$$$$$   /$$$$$$  /$$  /$$  /$$ /$$$$$$$ ")
        print(r"| $$  | $$| $$__  $$| $$  /$$/| $$__  $$ /$$__  $$| $$ | $$ | $$| $$__  $$")
        print(r"| $$  | $$| $$  \ $$| $$$$$$/ | $$  \ $$| $$  \ $$| $$ | $$ | $$| $$  \ $$")
        print(r"| $$  | $$| $$  | $$| $$_  $$ | $$  | $$| $$  | $$| $$ | $$ | $$| $$  | $$")
        print(r"|  $$$$$$/| $$  | $$| $$ \  $$| $$  | $$|  $$$$$$/|  $$$$$/$$$$/| $$  | $$")
        print(r" \______/ |__/  |__/|__/  \__/|__/  |__/ \______/  \_____/\___/ |__/  |__/")
        print("         a      l      g      o      r      i      t      h      m        ")
    print('')

###############################
##########  M A I N  ##########
###############################

print(r"                                                                     ")
print(r"                                                          ,----..    ")
print(r"    ,---,        ,---,        ,---,       ,---,          /   /   \   ")
print(r"  .'  .' `\    .'  .' `\    .'  .' `\    '  .' \        /   .     :  ")
print(r",---.'     \ ,---.'     \ ,---.'     \  /  ;    '.     .   /   ;.  \ ")
print(r"|   |  .`\  ||   |  .`\  ||   |  .`\  |:  :       \   .   ;   /  ` ; ")
print(r":   : |  '  |:   : |  '  |:   : |  '  |:  |   /\   \  ;   |  ; \ ; | ")
print(r"|   ' '  ;  :|   ' '  ;  :|   ' '  ;  :|  :  ' ;.   : |   :  | ; | ' ")
print(r"'   | ;  .  |'   | ;  .  |'   | ;  .  ||  |  ;/  \   \.   |  ' ' ' : ")
print(r"|   | :  |  '|   | :  |  '|   | :  |  ''  :  | \  \ ,''   ;  \; /  | ")
print(r"'   : | /  ; '   : | /  ; '   : | /  ; |  |  '  '--'   \   \  ',  /  ")
print(r"|   | '` ,/  |   | '` ,/  |   | '` ,/  |  :  :          ;   :    /   ")
print(r";   :  .'    ;   :  .'    ;   :  .'    |  | ,'           \   \ .'    ")
print(r"|   ,.'      |   ,.'      |   ,.'      `--''              `---`      ")
print(r"'---'oubly   '---'riven   '---'amped       nharmonic           scillator ")
# print("Doubly       Driven       Damped       Anharmonic     Oscillator     ")
print("                                                                     ")

print("                       ADRIEN POINDRON                               ")
print("                     University of Basel                             ")
print("                        January  2025                                ")

print("version 3 (March 14, 2025)")

print()
print('=== INSTANTIATE DDDAO CLASS ===')

solve_ivp_method = 'RK45' # LSODA
algorithm_print(solve_ivp_method)

f_z = 422500
# delta_f_z = 2000
# offset_f_z = -500
f_t = f_z # np.linspace(f_z+offset_f_z - delta_f_z, f_z+offset_f_z + delta_f_z, 41)

phi_nw = np.linspace(0, 2*math.pi, 41)

simulation = DDDAO_3(V_tkl = 0.025, V_nw = 1, V_piezo = 5/4.512899038958427, #  *0.7343421052582553   *0.9577642276422764
                        omega_z = f_z*2*math.pi, omega_drive = f_t*2*math.pi,
                        phi_tkl = 0, phi_nw = phi_nw,
                        B = -4.5e19*0,
                        cooling_rate = None, nl_damping_rate = None, #  345.8221951306219 0.03519716659973978
                        i_init = 5000, i_freq_sweep = 10000, i_smooth_sweep = 25,
                        sweep_length = len(phi_nw),
                        add_string=100)

start = time.time()
simulation.run_init()
simulation.run_freq_sweep()
simulation.run_end()
end = time.time()

print('=== SIMULATION COMPLETE ===')
print(f'Simulation duration : {end-start:.3f} s')
print(f'Simulation duration : {(end-start)/60:.3f} min')
print(f'Simulation duration : {(end-start)/60/60:.3f} h')
print('=== END OF EXECUTION ===')
print()