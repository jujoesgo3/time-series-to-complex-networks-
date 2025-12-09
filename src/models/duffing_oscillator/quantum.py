import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from qutip import destroy, Options, displace, basis

from .params import (
    omega_quantum, N, hbar, F,
    x0, p0, sigma_x, sigma_p,
    t_span, n_points
)


def build_operators():
    """
    Construye operadores cuánticos básicos.
    """
    a = destroy(N)
    x_op = (a + a.dag()) * np.sqrt(hbar / 2)
    p_op = -1j * (a - a.dag()) * np.sqrt(hbar / 2)
    return a, x_op, p_op


def build_hamiltonian(a):
    """
    Construye el Hamiltoniano dependiente del tiempo.
    """
    H_osc = hbar * (a.dag() * a + 0.5)
    H_nonlinear = (hbar**2 / 16) * (a.dag() + a) ** 4
    H_drive = -np.sqrt(hbar / 2) * (a.dag() + a) * F

    H = [
        H_osc + H_nonlinear,
        [H_drive, "np.cos(omega_quantum * t)"]
    ]
    return H


def initial_state():
    """
    Construye el estado inicial coherente.
    """
    alpha = (x0 + 1j * p0) / np.sqrt(2 * hbar)

    # Estado desplazado y comprimido (squeezed state) para ajustar anchos
    # (Requiere importar "squeeze" de qutip)
    from qutip import squeeze
    z = np.log(sigma_x / np.sqrt(hbar/2))  # Parámetro de compresión
    psi0 = displace(N, alpha) * squeeze(N, z) * basis(N, 0)
    return psi0

def simulate(plot=False):
    """
    Ejecuta la simulación cuántica del oscilador de Duffing.
    """
    a, x_op, p_op = build_operators()
    H = build_hamiltonian(a)
    psi0 = initial_state()
    psi0 = psi0 / psi0.norm()  # Normalizar
	
    tlist = np.linspace(*t_span, n_points)

    options = Options(
        nsteps=100000,
        max_step=0.01,
        method="lsoda"
    )

    result = qt.mesolve(
        H,
        psi0,
        tlist,
        c_ops=[],
        e_ops=[x_op, p_op],
        args={"omega_quantum": omega_quantum},
        options=options
    )

    x_t = result.expect[0]
    p_t = result.expect[1]

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        ax[0].plot(tlist, x_t)
        ax[0].set_title(r"$\langle x \rangle$")
        ax[0].set_xlabel("Tiempo")
        ax[0].set_ylabel("Posición")

        ax[1].plot(tlist, p_t)
        ax[1].set_title(r"$\langle p \rangle$")
        ax[1].set_xlabel("Tiempo")
        ax[1].set_ylabel("Momento")

        plt.tight_layout()
        plt.show()

    return tlist, x_t, p_t


if __name__ == "__main__":
    t, x_t, p_t = simulate(plot=True)
    print("Simulación cuántica de Duffing completada")

