import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from params import N, K, nbar, delta_n, tlist, data_file


def load_initial_data():
   
    df = pd.read_csv(data_file)
    omega = df["omega"].values
    theta0 = df["theta0_class"].values
    return omega, theta0


def dtheta_dn(y, t, omega, K, N, ntot):
    
    theta = y[:N]
    n = y[N:]

    z = np.sqrt(N / ntot) * np.sum(np.sqrt(n) * np.exp(1j * theta))
    r = np.abs(z)
    phi = np.angle(z)

    dtheta_dt = np.zeros(N)
    dn_dt = np.zeros(N)

    for j in range(N):
        if n[j] == 0:
            continue

        dtheta_dt[j] = omega[j] + (K / (2 * ntot)) * np.sum(
            np.sqrt(n[j] * n) * (3 - (n / n[j])) * np.sin(theta - theta[j])
        )

        dn_dt[j] = -(K / (2 * ntot)) * np.sum(
            2 * np.sqrt(n[j] * n) * (n - n[j]) * np.cos(theta - theta[j])
        )

    return np.concatenate([dtheta_dt, dn_dt])


def simulate(plot=False):
    
    omega, theta0 = load_initial_data()

    n0 = np.random.normal(loc=nbar, scale=delta_n, size=N)
    y0 = np.concatenate([theta0, n0])
    ntot = np.sum(n0)

    sol = odeint(
        dtheta_dn,
        y0,
        tlist,
        args=(omega, K, N, ntot)
    )

    theta_t = sol[:, :N]
    n_t = sol[:, N:]
    theta_q = np.unwrap(theta_t, axis=1)

    # Parámetro de orden
    r_semi = []
    angle_semi = []

    for n, theta in zip(n_t, theta_t):
        nt = n.sum()
        med = np.mean(n)
        z = (1 / N) * np.sum(np.sqrt(n / med) * np.exp(1j * theta))
        r_semi.append(np.abs(z))
        angle_semi.append(np.angle(z))

    r_semi = np.array(r_semi)
    angle_semi = np.array(angle_semi)

    if plot:
        # Evolución de ocupaciones
        plt.figure(figsize=(12, 4))
        plt.plot(tlist, n_t, alpha=0.4)
        plt.xlabel("Tiempo")
        plt.ylabel("n(t)")
        plt.show()

        # Comparación r(t)
        plt.figure(figsize=(12, 5))
        plt.plot(tlist, r_semi, label="Cuántico/Semiclásico", linestyle="--")
        plt.xlabel("Tiempo")
        plt.ylabel("Orden r")
        plt.legend()
        plt.show()

    return tlist, theta_q, n_t, r_semi, angle_semi


if __name__ == "__main__":
    simulate(plot=True)
    print("Simulación cuántica/semiclásica de Kuramoto completada")


