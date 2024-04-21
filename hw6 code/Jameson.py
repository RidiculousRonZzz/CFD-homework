import numpy as np

def jameson(U0, F0, gamma, k2, k4, dx, dt, Mx, t0, Time, time_out):
    # Initialize variables
    U = U0.copy()
    F = F0.copy()
    F_p = F0.copy()
    F_m = F0.copy()
    len_out = len(time_out)
    U_out = np.zeros((3, Mx + 1, len_out))
    num_out = 0
    current_time = t0
    out_idx = 0

    while current_time <= Time:
        rho = U[0, :]
        u = U[1, :] / U[0, :]
        p = (gamma - 1) * (U[2, :] - rho * u**2 / 2)
        a = np.sqrt(gamma * np.abs(p) / np.abs(rho))
        F[0, :] = U[1, :]
        F[1, :] = rho * u**2 + p
        F[2, :] = (U[2, :] + p) * u

        lam_max = np.abs(u) + a
        lam_max_p = 0.5 * (lam_max + np.roll(lam_max, -1))
        lam_max_m = 0.5 * (lam_max + np.roll(lam_max, 1))

        v = np.abs(np.roll(p, -1) - 2 * p + np.roll(p, 1)) / np.abs(np.roll(p, -1) + 2 * p + np.roll(p, 1))
        eps2_p = k2 * np.max(np.vstack([np.roll(v, 2), np.roll(v, 1), v, np.roll(v, -1)]), axis=0)
        eps2_m = k2 * np.max(np.vstack([np.roll(v, 1), v, np.roll(v, -1), np.roll(v, -2)]), axis=0)

        eps4_p = k4 - eps2_p
        eps4_m = k4 - eps2_m
        eps4_p = (eps4_p + np.abs(eps4_p)) / 2
        eps4_m = (eps4_m + np.abs(eps4_m)) / 2

        F_p = 0.5 * (F + np.roll(F, -1)) - eps2_p * lam_max_p * (np.roll(U, -1) - U) \
              + eps4_p * lam_max_p * (np.roll(U, -2) - 3 * np.roll(U, -1) + 3 * U - np.roll(U, 1))
        F_m = 0.5 * (F + np.roll(F, 1)) - eps2_m * lam_max_m * (U - np.roll(U, 1)) \
              + eps4_m * lam_max_m * (np.roll(U, 1) - 3 * U + 3 * np.roll(U, 1) - np.roll(U, 2))

        # Boundary conditions
        F_p[:, 0] = F0[:, 0]
        F_p[:, -1] = F0[:, -1]
        F_m[:, 0] = F0[:, 0]
        F_m[:, -1] = F0[:, -1]

        U = U - dt / dx * (F_p - F_m)

        # Storing results at specified times
        if num_out < len_out and current_time >= time_out[num_out]:
            U_out[:, :, num_out] = U
            time_out[num_out] = current_time
            num_out += 1
        
        current_time += dt

    return U_out, time_out

# Example initialization of parameters and call to function (you need to define these variables appropriately)
# U0, F0, gamma, k2, k4, dx, dt, Mx, t0, Time, time_out = ...
# U_out, time_out = jameson(U0, F0, gamma, k2, k4, dx, dt, Mx, t0, Time, time_out)
