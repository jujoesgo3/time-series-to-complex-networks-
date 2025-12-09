import numpy as np
import os

# ---------------------------
# Parámetros generales
# ---------------------------
N = 50
K = 1.8
delta_omega = 1.0
nbar = 5.0
delta_n = 0.1

# ---------------------------
# Parámetros temporales
# ---------------------------
dt = 0.05
t_max = 260.0
tlist = np.arange(0, t_max + dt, dt)

# ---------------------------
# Ruta de datos
# ---------------------------
BASE_DIR = os.path.dirname(__file__)
data_file = os.path.join(BASE_DIR, "datos_omega_theta.csv")


