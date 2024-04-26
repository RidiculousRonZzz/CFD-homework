import numpy as np
import matplotlib.pyplot as plt

velocity = 1.0  
domain_length = 1.0  
num_points = 100  
courant_num = 0.5  
critical_times = [0.1, 1.0, 10.0]  

delta_x = domain_length / num_points  
delta_t = courant_num * delta_x / velocity  

def minmod_limiter(value1, value2):
    if value1 * value2 > 0:
        return np.sign(value1) * min(abs(value1), abs(value2))
    else:
        return 0.0

def wave_initial_condition(position):
    if -0.25 <= position <= 0.25:
        return 1.0
    return 0.0

positions = np.linspace(-0.5 + delta_x/2, 0.5 - delta_x/2, num_points)
wave_values = np.array([wave_initial_condition(pos) for pos in positions])
wave_history = [wave_values.copy()]

for current_time in np.arange(0, max(critical_times), delta_t):
    new_wave_values = wave_values.copy()
    for index in range(1, num_points-1):
        slope = minmod_limiter((wave_values[index] - wave_values[index-1])/delta_x, (wave_values[index+1] - wave_values[index])/delta_x)
        prev_slope = minmod_limiter((wave_values[index-1] - wave_values[index-2])/delta_x, (wave_values[index] - wave_values[index-1])/delta_x)
        
        new_wave_values[index] = wave_values[index] - (delta_t/delta_x) * (velocity * (wave_values[index] - wave_values[index-1]) + velocity * delta_x * (slope - prev_slope)/2) + (delta_t**2/delta_x) * velocity**2 * (slope - prev_slope)/2
    
    new_wave_values[0] = new_wave_values[-2]
    new_wave_values[-1] = new_wave_values[1]
    
    wave_values = new_wave_values.copy()
    if np.any([np.isclose(current_time + delta_t, t_point, atol=delta_t/2) for t_point in critical_times]):
        wave_history.append(wave_values.copy())

plt.plot(positions, wave_history[0], label='t = 0.1')
plt.plot(positions, wave_history[1], label='t = 1.0')
plt.plot(positions, wave_history[2], label='t = 10.0')
plt.xlabel('Position (x)')
plt.ylabel('Wave amplitude (u)')
plt.legend()
plt.show()
