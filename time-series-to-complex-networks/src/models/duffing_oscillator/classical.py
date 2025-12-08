import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from .params import alpha, F, omega, y0, t_span, n_points


def duffing_autonomous(t, state):
  
    x, y = state
    dxdt = y
    dydt = -x - x**3 + F * np.cos(omega * t)
    return [dxdt, dydt]


def simulate(plot=False):
  
    t_eval = np.linspace(*t_span, n_points)
    sol = solve_ivp(
        duffing_autonomous,
        t_span,
        y0,
        t_eval=t_eval
    )

    x = sol.y[0]
    y = sol.y[1]

    if plot:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(x, y, linewidth=0.5, label="Atractor")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        plt.show()

    return sol.t, sol.y


if __name__ == "__main__":
    
    t, states = simulate(plot=True)
    print("Simulación de Duffing completada")

