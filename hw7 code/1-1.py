import numpy as np
import matplotlib.pyplot as plt

specific_heat_ratio = 1.4
simulation_time = 0.2

def initial_state(x_position):
    if x_position < 0.5:
        return np.array([1.0, 0.0, 1.0])
    else:
        return np.array([0.125, 0.0, 0.1])

def state_to_conservative(primitive_state):
    density, velocity, pressure = primitive_state
    energy = pressure / (specific_heat_ratio - 1) + 0.5 * density * velocity**2
    return np.array([density, density * velocity, energy])

def conservative_to_state(conservative_state):
    density, momentum, total_energy = conservative_state
    velocity = momentum / density
    pressure = (total_energy - 0.5 * momentum**2 / density) * (specific_heat_ratio - 1)
    return np.array([density, velocity, pressure])

def calculate_flux(conservative_state):
    density, momentum, total_energy = conservative_state
    velocity = momentum / density
    pressure = (total_energy - 0.5 * momentum**2 / density) * (specific_heat_ratio - 1)
    return np.array([momentum, momentum**2 / density + pressure, (total_energy + pressure) * velocity])

def roe_avg(left_state, right_state):
    left = conservative_to_state(left_state)
    right = conservative_to_state(right_state)
    
    density_left, velocity_left, pressure_left = left
    density_right, velocity_right, pressure_right = right
    
    enthalpy_left = specific_heat_ratio * pressure_left / (density_left * (specific_heat_ratio - 1)) + 0.5 * velocity_left**2
    enthalpy_right = specific_heat_ratio * pressure_right / (density_right * (specific_heat_ratio - 1)) + 0.5 * velocity_right**2

    sqrt_density_left = np.sqrt(density_left)
    sqrt_density_right = np.sqrt(density_right)
    
    averaged_density = sqrt_density_left * sqrt_density_right
    averaged_velocity = (sqrt_density_left * velocity_left + sqrt_density_right * velocity_right) / (sqrt_density_left + sqrt_density_right)
    averaged_enthalpy = (sqrt_density_left * enthalpy_left + sqrt_density_right * enthalpy_right) / (sqrt_density_left + sqrt_density_right)
    sound_speed_squared = (specific_heat_ratio - 1) * (averaged_enthalpy - 0.5 * averaged_velocity**2)
    
    if sound_speed_squared < 0:
        raise ValueError("Negative sound speed squared.")

    sound_speed = np.sqrt(sound_speed_squared)
    
    return {'density': averaged_density, 'velocity': averaged_velocity, 'enthalpy': averaged_enthalpy, 'sound_speed': sound_speed}

def eigens_calc(roe_avg_state):
    velocity_hat = roe_avg_state['velocity']
    enthalpy_hat = roe_avg_state['enthalpy']
    sound_speed_hat = roe_avg_state['sound_speed']
    
    eigenvalues = np.array([velocity_hat - sound_speed_hat, velocity_hat, velocity_hat + sound_speed_hat])
    
    right_eigenvectors = np.array([
        [1, velocity_hat - sound_speed_hat, enthalpy_hat - velocity_hat * sound_speed_hat],
        [1, velocity_hat, 0.5 * velocity_hat**2],
        [1, velocity_hat + sound_speed_hat, enthalpy_hat + velocity_hat * sound_speed_hat]
    ])
    
    gamma_alpha = 0.5 * (specific_heat_ratio - 1) / sound_speed_hat**2
    left_eigenvectors = np.array([
        [0.5 + gamma_alpha * velocity_hat**2, 
         - (gamma_alpha * specific_heat_ratio * velocity_hat + 0.5 / sound_speed_hat), 
         gamma_alpha * specific_heat_ratio - 1],
        [1 - gamma_alpha * velocity_hat**2, 
         specific_heat_ratio * gamma_alpha * velocity_hat, 
         - specific_heat_ratio * gamma_alpha],
        [0.5 + gamma_alpha * velocity_hat**2, 
         - (gamma_alpha * velocity_hat / specific_heat_ratio - 0.5 / sound_speed_hat), 
         gamma_alpha - 0.5 / sound_speed_hat**2]
    ])
    
    return eigenvalues, right_eigenvectors, left_eigenvectors

def state_difference(left, right):
    return right - left

def additional_flux_calc(left_state, right_state, epsilon=1e-1):
    roe_state = roe_avg(left_state, right_state)
    density = roe_state['density']
    sound_speed = roe_state['sound_speed']
    velocity = roe_state['velocity']
    eigenvalues, right_eigenvectors, _ = eigens_calc(roe_state)
    delta_U = state_difference(left_state, right_state)
    left_primitive = conservative_to_state(left_state)
    right_primitive = conservative_to_state(right_state)
    delta_primitive = right_primitive - left_primitive
    delta_density = delta_primitive[0]
    delta_velocity = delta_primitive[1]
    delta_pressure = delta_primitive[2]
    
    beta = np.zeros(3)
    beta[0] = 0.5 * (delta_pressure - density * sound_speed * delta_velocity) / (sound_speed**2)
    beta[1] = delta_density - delta_pressure / (sound_speed**2)
    beta[2] = 0.5 * (delta_pressure + density * sound_speed * delta_velocity) / (sound_speed**2)
    
    additional_flux = beta[0] * right_eigenvectors[0] * np.abs(eigenvalues[0]) + beta[1] * right_eigenvectors[1] * np.abs(eigenvalues[1]) + beta[2] * right_eigenvectors[2] * np.abs(eigenvalues[2])
    
    return additional_flux

def roe_numerical_flux(U, index, dx):
    UL = U[index]
    UR = U[index + 1]
    
    roe_state = roe_avg(UL, UR)
    rho = roe_state['density']
    a = roe_state['sound_speed']
    u = roe_state['velocity']
    
    flux_result = np.zeros(3)
    flux_result = 0.5 * (calculate_flux(UL) + calculate_flux(UR)) - 0.5 * additional_flux_calc(UL, UR)
    
    return flux_result

def simulation_flux_calc(num_cells, simulation_time):
    dx = 1.0 / (num_cells - 1)
    dt = 0.0001
    x_values = np.linspace(0, 1, num_cells)
    U = np.array([state_to_conservative(initial_state(xi)) for xi in x_values])
    current_time = 0
    while current_time < simulation_time:
        flux_values = np.zeros_like(U)
        for j in range(3, num_cells - 3):
            flux_values[j] = roe_numerical_flux(U, j, dx)
        
        for j in range(5, num_cells - 5):
            U[j] -= dt/dx * (flux_values[j] - flux_values[j - 1])
   
        current_time += dt
    
    return x_values, np.array([conservative_to_state(Ui) for Ui in U])

def plot_results(nx, t_final):
        x, results = simulation_flux_calc(nx, t_final)
        nx = nx - 1
        plt.figure(figsize=(11, 4))
        plt.subplot(1, 3, 1)
        plt.plot(x, results[:, 0], label=f'nx={nx}')
        plt.subplot(1, 3, 2)
        plt.plot(x, results[:, 1], label=f'nx={nx}')
        plt.subplot(1, 3, 3)
        plt.plot(x, results[:, 2], label=f'nx={nx}')
        plt.subplot(1, 3, 1)
        plt.title('Density')
        plt.xlabel('x')
        plt.subplot(1, 3, 2)
        plt.title('Velocity')
        plt.xlabel('x')
        plt.subplot(1, 3, 3)
        plt.title('Pressure')
        plt.xlabel('x')
        plt.tight_layout()
        plt.show()
    
plot_results(201, 0.2)
