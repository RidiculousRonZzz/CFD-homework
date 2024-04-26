import numpy as np
import matplotlib.pyplot as plt

alpha = 1.4
time_limit = 0.2
small_value = 0.001

def initial_setup(position):
    if position < 0.5:
        return np.array([5, np.sqrt(alpha), 29.0])
    else:
        return np.array([1.0, 5*np.sqrt(alpha), 1.0])

def to_conservative(state_array):
    density, velocity, pressure = state_array
    energy = pressure / (alpha - 1) + 0.5 * density * velocity**2
    return np.array([density, density * velocity, energy])

def to_primitive(conserved_array):
    density, momentum, energy = conserved_array
    velocity = momentum / density
    pressure = (energy - 0.5 * momentum**2 / density) * (alpha - 1)
    return np.array([density, velocity, pressure])

def compute_flux(conserved_values):
    density, momentum, energy = conserved_values
    velocity = momentum / density
    pressure = (energy - 0.5 * momentum**2 / density) * (alpha - 1)
    return np.array([momentum, momentum**2 / density + pressure, (energy + pressure) * velocity])

def average_states(left, right):
    left_prim = to_primitive(left)
    right_prim = to_primitive(right)
    
    density_L, velocity_L, pressure_L = left_prim
    density_R, velocity_R, pressure_R = right_prim
    
    enthalpy_L = alpha * pressure_L / (density_L * (alpha - 1)) + 0.5 * velocity_L**2
    enthalpy_R = alpha * pressure_R / (density_R * (alpha - 1)) + 0.5 * velocity_R**2

    sqrt_density_L = np.sqrt(density_L)
    sqrt_density_R = np.sqrt(density_R)
    
    rho_avg = sqrt_density_L * sqrt_density_R
    velocity_avg = (sqrt_density_L * velocity_L + sqrt_density_R * velocity_R) / (sqrt_density_L + sqrt_density_R)
    enthalpy_avg = (sqrt_density_L * enthalpy_L + sqrt_density_R * enthalpy_R) / (sqrt_density_L + sqrt_density_R)
    a_squared = (alpha - 1) * (enthalpy_avg - 0.5 * velocity_avg**2)
    
    if a_squared < 0:
        raise ValueError("Negative sound speed squared.")

    sound_speed = np.sqrt(a_squared)

    average_state = {
        'density': rho_avg,
        'velocity': velocity_avg,
        'enthalpy': enthalpy_avg,
        'sound_speed': sound_speed
    }
    
    return average_state

def eigenvalues_vectors(average):
    velocity_hat = average['velocity']
    enthalpy_hat = average['enthalpy']
    sound_speed_hat = average['sound_speed']
    
    lambdas = np.array([velocity_hat - sound_speed_hat, velocity_hat, velocity_hat + sound_speed_hat])
    
    right_vecs = np.array([
        [1, velocity_hat - sound_speed_hat, enthalpy_hat - velocity_hat * sound_speed_hat],
        [1, velocity_hat, 0.5 * velocity_hat**2],
        [1, velocity_hat + sound_speed_hat, enthalpy_hat + velocity_hat * sound_speed_hat]
    ])
    
    alpha_factor = 0.5 * (alpha - 1) / sound_speed_hat**2
    left_vecs = np.array([
        [0.5 + alpha_factor * velocity_hat**2, 
         - (alpha_factor * alpha * velocity_hat + 0.5 / sound_speed_hat), 
         alpha_factor * alpha - 1],
        [1 - alpha_factor * velocity_hat**2, 
         alpha * alpha_factor * velocity_hat, 
         - alpha * alpha_factor],
        [0.5 + alpha_factor * velocity_hat**2, 
         - (alpha_factor * velocity_hat / alpha - 0.5 / sound_speed_hat), 
         alpha_factor - 0.5 / sound_speed_hat**2]
    ])
    
    return lambdas, right_vecs, left_vecs

def state_difference(UL, UR):
    return UR - UL

def compute_flux_correction(UL, UR):
    avg_state = average_states(UL, UR)
    eigenvals, right_eigvecs, left_eigvecs = eigenvalues_vectors(avg_state)
    delta_state = state_difference(UL, UR)
    delta_prim = state_difference(to_primitive(UL), to_primitive(UR))
    delta_dens, delta_vel, delta_pres = delta_prim
    
    beta = np.zeros(3)
    beta[0] = 0.5 * (delta_pres - avg_state['density'] * avg_state['sound_speed'] * delta_vel) / (avg_state['sound_speed']**2)
    beta[1] = delta_dens - delta_pres / (avg_state['sound_speed']**2)
    beta[2] = 0.5 * (delta_pres + avg_state['density'] * avg_state['sound_speed'] * delta_vel) / (avg_state['sound_speed']**2)
    
    for i in range(3):
        if(np.abs(eigenvals[i]) < small_value):
            eigenvals[i] = (eigenvals[i]**2 + small_value**2) / (2 * small_value)
    
    corrected_flux = np.sum([beta[i] * right_eigvecs[i] * abs(eigenvals[i]) for i in range(3)], axis=0)
    
    return corrected_flux

def roe_solver(states, index, dx):
    left_state = states[index]
    right_state = states[index + 1]
    
    flux_average = 0.5 * (compute_flux(left_state) + compute_flux(right_state)) - 0.5 * compute_flux_correction(left_state, right_state)
    return flux_average

def simulate_flux(nx, final_time):
    dx = 1.0 / (nx - 1)
    dt = 0.0001
    positions = np.linspace(0, 1, nx)
    states = np.array([to_conservative(initial_setup(x)) for x in positions])
    time = 0
    while time < final_time:
        fluxes = np.zeros_like(states)
        for j in range(3, nx - 3):
            fluxes[j] = roe_solver(states, j, dx)
        
        for j in range(5, nx - 5):
            states[j] -= dt / dx * (fluxes[j] - fluxes[j - 1])
   
        time += dt

    return positions, np.array([to_primitive(state) for state in states])

def display_results(nx, end_time):
        x, results = simulate_flux(nx, end_time)
        nx = nx - 1

        plt.figure(figsize=(11, 4))

        plt.subplot(1, 3, 1)
        plt.plot(x, results[:, 0], label=f'nx={nx}')
        plt.subplot(1, 3, 2)
        plt.plot(x, results[:, 1], label=f'nx={nx}')
        plt.subplot(1, 3, 3)
        plt.plot(x, results[:, 2], label=f'nx={nx}')
        plt.subplot(1, 3, 1)
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.subplot(1, 3, 2)
        plt.title(f'epsilon={small_value}')
        plt.xlabel('x')
        plt.ylabel('Velocity')
        plt.subplot(1, 3, 3)
        plt.xlabel('x')
        plt.ylabel('Pressure')
        plt.tight_layout()
        plt.show()

display_results(201, 0.2)
