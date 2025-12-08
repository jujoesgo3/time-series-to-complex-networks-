import numpy as np

# Parámetros físicos
alpha = 1.0
F = 1.5 / np.sqrt(0.3) ** 3
omega = 0.325


omega_quantum = 0.325
hbar=1
sigma_x = np.sqrt(hbar/2)      # Ancho de la Gaussiana en posición
sigma_p = np.sqrt(hbar/2)      # Ancho de la Gaussiana en momento
N=100
x0 = hbar/4
p0 = 0.0 

# Condiciones iniciales
y0 = (1/4, 0.0)

# Parámetros numéricos
t_span = (0.0, 500.0)
n_points = 5000

