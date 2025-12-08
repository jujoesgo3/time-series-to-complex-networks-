import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from params import K, tlist, data_file


def load_initial_data():
   
    df = pd.read_csv(data_file)
    omega = df["omega"].values
    theta0 = df["theta0_class"].values
    return omega, theta0


def dtheta_class(theta, t, omega, K):
   
    z = np.mean(np.exp(1j * theta))
    r = np.abs(z)
    phi = np.angle(z)
    return omega + K * r * np.sin(phi - theta)


def simulate(plot=False):
   
    omega, theta0 = load_initial_data()

    sol = odeint(
        dtheta_class,
        theta0,
        tlist,
        args=(omega, K)
    )

    theta_t = np.unwrap(sol, axis=1)
    r_class = np.abs(np.mean(np.exp(1j * theta_t), axis=1))

    if plot:
        plt.figure(figsize=(12, 5))
        plt.plot(tlist, r_class, label="Clásico")
        plt.xlabel("Tiempo")
        plt.ylabel("Orden r")
        plt.legend()
        plt.show()

    return tlist, theta_t, r_class


if __name__ == "__main__":
    t, theta, r = simulate(plot=True)
    print("Simulación clásica de Kuramoto completada")

