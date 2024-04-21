import numpy as np
import matplotlib.pyplot as plt
from Rusanov import rusanov
from Lax import lax
from Jameson import jameson

# Constants
gamma = 1.4
xRange = [-0.5, 0.5]
time_out = np.array([0.1, 0.2, 0.25])
dt = 0.00005
t0 = 0
Time = 0.35

# Numerical methods to choose from
methods = {
    'Rusanov': rusanov,
    'Lax': lax,
    'Jameson': jameson
}

# Function to run simulation and plot for different Mx values
def run_simulation(Mx, method_name):
    dx = (xRange[1] - xRange[0]) / Mx
    x = np.linspace(xRange[0], xRange[1], Mx + 1)

    # Initial conditions
    rho0 = np.ones(Mx + 1) * 0.125
    u0 = np.zeros(Mx + 1)
    p0 = np.ones(Mx + 1) * 0.1
    rho0[:Mx//2 + 1] = 1.0
    u0[:Mx//2 + 1] = 0.75
    p0[:Mx//2 + 1] = 1.0

    U0 = np.zeros((3, Mx + 1))
    U0[0, :] = rho0
    U0[1, :] = rho0 * u0
    U0[2, :] = p0 / (gamma - 1) + rho0 * u0**2 / 2

    F0 = np.zeros((3, Mx + 1))
    F0[0, :] = U0[1, :]
    F0[1, :] = rho0 * u0**2 + p0
    F0[2, :] = (U0[2, :] + p0) * u0

    # Choose method
    if method_name == 'Jameson':
        k2 = 0.5
        k4 = 0.5 / 2
        U_out, time_out_returned = methods[method_name](U0, F0, gamma, k2, k4, dx, dt, Mx, t0, Time, time_out)
    else:
        U_out, time_out_returned = methods[method_name](U0, F0, gamma, dx, dt, Mx, t0, Time, time_out)

    # Extract variables from U_out
    rho_out = U_out[0, :, :]
    u_out = U_out[1, :, :] / rho_out
    p_out = (gamma - 1) * (U_out[2, :, :] - 0.5 * rho_out * u_out**2)

    # Plotting results
    initial_vars = [u0, rho0, p0]
    result_vars = [u_out, rho_out, p_out]
    variable_labels = ['Velocity (u)', 'Density (ρ)', 'Pressure (p)']
    titles = ['Velocity Profile', 'Density Profile', 'Pressure Profile']

    for ini_var, res_var, label, title in zip(initial_vars, result_vars, variable_labels, titles):
        plt.figure(figsize=(10, 6))
        plt.plot(x, ini_var, label='Initial (t=0s)', linestyle='dashed', color='gray', linewidth=1.5)
        for i, t in enumerate(time_out):
            plt.plot(x, res_var[:, i], label=f't={t:.2f}s', linewidth=1.5)
        plt.title(f'{method_name} Method - {title} - Mx = {Mx}', fontsize=20)
        plt.legend(loc='upper left')
        plt.xlabel('x')
        plt.ylabel(label)
        plt.grid(True)
        plt.show()

# Loop over different grid sizes
grid_sizes = [100, 500, 1000]
for Mx in grid_sizes:
    run_simulation(Mx, 'Rusanov')  # Change 'Lax' to any other method as needed
