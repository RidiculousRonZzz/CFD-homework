import numpy as np

def lax(U0, F0, gamma, dx, dt, Mx, t0, Time, time_out):
    # Initialization
    U = U0.copy()
    F = F0.copy()
    F_p = F0.copy()
    F_m = F0.copy()
    len_out = len(time_out)
    U_out = np.zeros((3, Mx + 1, len_out))
    num_out = 0
    current_time = t0
    
    # Main simulation loop
    while current_time <= Time:
        rho = U[0, :]
        u = U[1, :] / rho
        p = (gamma - 1) * (U[2, :] - 0.5 * rho * u**2)
        a = np.sqrt(gamma * np.abs(p) / np.abs(rho))
        F[0, :] = U[1, :]
        F[1, :] = rho * u**2 + p
        F[2, :] = (U[2, :] + p) * u

        lam_max = np.abs(u) + a
        lam_max_p = 0.5 * (lam_max + np.roll(lam_max, -1))
        lam_max_m = 0.5 * (lam_max + np.roll(lam_max, 1))

        eps_p = 0.5 * dx / dt / lam_max_p
        eps_m = 0.5 * dx / dt / lam_max_m

        F_p = 0.5 * (F + np.roll(F, -1)) - eps_p * lam_max_p * (np.roll(U, -1) - U)
        F_m = 0.5 * (F + np.roll(F, 1)) - eps_m * lam_max_m * (U - np.roll(U, 1))

        # Applying boundary conditions
        F_p[:, 0] = F0[:, 0]
        F_p[:, -1] = F0[:, -1]
        F_m[:, 0] = F0[:, 0]
        F_m[:, -1] = F0[:, -1]

        U = U - dt / dx * (F_p - F_m)

        # Output data at specified times
        if num_out < len_out and current_time >= time_out[num_out]:
            U_out[:, :, num_out] = U
            time_out[num_out] = current_time
            num_out += 1
        
        current_time += dt

    return U_out, time_out

# Example of parameters (you'll need to define these properly with actual values)
# U0, F0, gamma, dx, dt, Mx, t0, Time, time_out = ...
# U_out, time_out = lax(U0, F0, gamma, dx, dt, Mx, t0, Time, time_out)
