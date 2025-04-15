# Codes for REIN_TRAP_COOL

Programs used in the context of the REIN experiment (nanoWire) in Stefan Willitsch's group at Basel university. Some important programs to highlight are indicated with a $\star$ in this document.

The main programs of this repository are devoted to the solution of the modified Duffing equation accounting for a Doppler-cooled trapped ion in a linear with anharmonicities and non-linear friction.

## Theory

The modified Duffing equation is :

$$
\ddot{x} + 2\mu\dot{x} + \gamma\dot{x}^3 + \omega_0^2x + \alpha x^3 = k_{tkl}\cos{(\omega_{tkl} t + \varphi_{tkl})} + k_{nw}\cos{(\omega_{nw} t + \varphi_{nw})}
$$

with solution for the amplitude as :

$$
\frac{9}{16}(\alpha^2 + \gamma^2\omega_0^6) a^6 + 3\omega_0(\mu\gamma\omega_0^3 - \sigma\alpha)a^4 + 4\omega_0^2(\sigma^2 + \mu^2)a^2 - k^2 = 0.
$$

and the phase as

$$
\tan{\varphi} = \frac{8\mu\omega_0}{3\alpha a^2 - 8\omega_0\sigma}.
$$

_Note : this formula can be found in Akerman et al. article with a mistake in the right part of the denominator._

The scattering rate due to Doppler cooling can be written as :

$$
R_s(v) = \frac{\Gamma}{2}s_0 \times \frac{1}{1 + s_0 + 4(\frac{\delta(v)}{\Gamma})^2} = \frac{\Gamma}{2}s_0R_s^{_0}(v).
$$
After a third-order Taylor expansion it can be rewritten as
$$
\begin{align*}
R_s(v) &=s_0 R_s^{_0}({\footnotesize v=0})\\
&+\frac{4\delta_0}{\Gamma}s_0R_s^{_0}({\footnotesize v=0})^2 {\footnotesize\times}\; kv \\
&- \frac{2}{\Gamma}\left(1 + s_0 - 12\left(\frac{\delta_0}{\Gamma}\right)^2\right)s_0R_s^{_0}({\footnotesize v=0})^3 {\footnotesize\times}\; k^2v^2 \\
&- \frac{32\delta_0}{\Gamma^3}\left(1 + s_0 - 4(\frac{\delta_0}{\Gamma})^2\right)s_0R_s^{_0}({\footnotesize v=0})^4 {\footnotesize\times}\; k^3v^3 \\
&+ \mathcal{O}(v^4).
\end{align*}
$$

with $\delta(v) = \omega_l - \omega_0 -kv = \delta_0 - kv$, $s_0 = I/I_{sat}$, $I_{sat} = \hbar\omega_0^3/(12\pi c^3)\Gamma$.


## Folders
* `Analysis`
* `Images`
* List of contributors

## Programs

$\star$ `Doubly_driven_oscillator.py` (later `DDDAO_3`) is the main program of this repository. It solves the modified Duffing equation using `solve_ivp` function from `scipy` module. The physics is based upon the work of N. Akerman et al., "[Single-ion nonlinear mechanical oscillator](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.82.061402)," PRA 82, (2010).

There are several versions of this program :
- `Doubly_driven_oscillator_solveivp_1.py` : DO NOT USE THIS ONE. Solves the equation with the physical parameters explicitely set in the class. You can sweep the drive frequency of both tickle and nw with a common phase. In theory identical to `Doubly_driven_oscillator_solveivp_2.py` except for `self.time_switch` and improved `print()` functions.

- `Doubly_driven_oscillator_solveivp_2.py` : TESTED. Solves the equation with the physical parameters explicitely set in the class. You can sweep the drive frequency of both tickle and nw with a common phase. Added a `self.time_switch` to save when the frequency is changed.

- $\star$ `Doubly_driven_oscillator_solveivp_3` (then `DDDAO_3`) : Solves the equation for a range of parameters sweeped during an experiment. This allows to modify several parameters at once for each step of the solution.

$\star$ `Duffing_analytical_REIN.py` is a program solving the analytical solution of Duffing equation for the amplitude and phase. This is in the context of the REIN experiment. See the 'Akerman' version for the one based on the paper.

![Amplitude vs. detuning](Images/Duffing_roots_vs_detuning.png)

$\star$ `Taylor_expansion.ipynb` is a program solving the Taylor expansion of scattering rate of a two-level system (Doppler cooling).

`calcium_constants.py` is a set of general physical constant and $^{40}$ Ca related constants. An example of how to use it in your own python program is as follows : 
```python
import sys
sys.path.append("Relative/Path/To/Funcion/File")

import fit_functions as ff
from calcium_constants import m_Ca, C_e, kb, hbar, eps0, Coul_factor
```

`job_start.sh` is a bash program to execute the main python program `DDDAO_3.py` on a slurm cluster (Studix). You put this `.sh` in the same folder as the `.py` on the cluster and then run `sbatch job_start.sh`.

## `Analysis`
`YYMMDD_Doubly_driven_oscillator_testRK4.ipynb` analyse the output of the program `Doubly_driven_oscillator.py`. You have folders called `old_x`, x being an integer refering to the version of the `.py` program.

## List of contributors
* Adrien Poindron
