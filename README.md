Programs used to numerically solve the "modified" Duffing equation, in the context of the [REIN experiment](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.133.223201) (nanoWire) in Stefan Willitsch's group at Basel university.

## Theory

Following [this article](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.82.061402), the "modified" Duffing equation is :

$$
\ddot{x} + 2\mu\dot{x} + \gamma\dot{x}^3 + \omega_0^2x + \alpha x^3 = k\cos{\omega t}
$$

with solution for the amplitude as :

$$
\frac{9}{16}(\alpha^2 + \gamma^2\omega_0^6) a^6 + 3\omega_0(\mu\gamma\omega_0^3 - \sigma\alpha)a^4 + 4\omega_0^2(\sigma^2 + \mu^2)a^2 - k^2 = 0.
$$

## Program

`Doubly_driven_oscillator_solveivp.py` is a program solving numerically the "modified" Duffing equation.
The numerical solution relies on the function `solve_ivp` from `scipy` module. The method used is by default `RK45` ans `LSODA` proved to be a bit less precise. It is much faster that "naïvely" implementing a RK4 method by a factor 25 approximately.

![Amplitude vs. detuning](hilbert_height_250204_REINMAN_RK45_a0241_4_1040.png)
