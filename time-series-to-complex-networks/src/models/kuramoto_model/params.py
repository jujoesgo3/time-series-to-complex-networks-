import numpy as np

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
data_file = "datos_omega_theta.csv"

