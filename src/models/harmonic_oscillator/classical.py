import numpy as np
from scipy.integrate import solve_ivp
from .params import m, omega, t_span, n_points, y0


def hamiltonian_system(t, y):

    x, p = y
    dxdt = p / m
    dpdt = -m * omega**2 * x
    return [dxdt, dpdt]


def simulate():

    t_eval = np.linspace(*t_span, n_points)

    sol = solve_ivp(
        hamiltonian_system,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval
    )
    return sol.t, sol.y


if __name__ == "__main__":
    t, y = simulate()
    print("Simulación completada")

    
