import numpy as np
import matplotlib.pyplot as plt

gamma = 1.4  
t_final = 0.05  

def initial_state(x):
    if x < 0.5:
        return np.array([1.0, 0.0, 1.0])  
    else:
        return np.array([0.125, 0.0, 0.1])  

def flux(U):
    rho, m, E = U
    u = m / rho
    p = (E - 0.5 * m**2 / rho) * (gamma - 1)
    return np.array([m, m**2 / rho + p, (E + p) * u])

def roe_avg_state(UL, UR, gamma=1.4):

    UL = conservative_to_primitive(UL)
    UR = conservative_to_primitive(UR)
    
    rhoL, uL, pL = UL
    rhoR, uR, pR = UR
    
    HL = gamma*pL / (rhoL * (gamma - 1)) + 0.5 * uL**2
    HR = gamma*pR / (rhoR * (gamma - 1)) + 0.5 * uR**2

    sqrt_rhoL = np.sqrt(rhoL)
    sqrt_rhoR = np.sqrt(rhoR)
    
    roe_rho = sqrt_rhoL * sqrt_rhoR
    roe_u = (sqrt_rhoL * uL + sqrt_rhoR * uR) / (sqrt_rhoL + sqrt_rhoR)
    roe_H = (sqrt_rhoL * HL + sqrt_rhoR * HR) / (sqrt_rhoL + sqrt_rhoR)
    roe_a2 = (gamma - 1) * (roe_H - 0.5 * roe_u**2)

    roe_a = np.sqrt(roe_a2)

    roe_avg = {
        'rho': roe_rho,
        'u': roe_u,
        'H': roe_H,
        'a': roe_a
    }
    
    return roe_avg

def primitive_to_conservative(W):
    rho, u, p = W
    e = p / (gamma - 1) + 0.5 * rho * u**2
    return np.array([rho, rho * u, e])

def conservative_to_primitive(U):
    rho, m, E = U
    u = m / rho
    p = (E - 0.5 * m**2 / rho) * (gamma - 1)
    return np.array([rho, u, p])

def eigenvalues_and_vectors(roe_avg, gamma=1.4):

    u_hat = roe_avg['u']
    H_hat = roe_avg['H']
    a_hat = roe_avg['a']
    
    lambda_ = np.array([u_hat - a_hat, u_hat, u_hat + a_hat])
    
    r_vectors = np.array([
        [1, u_hat - a_hat, H_hat - u_hat * a_hat],
        [1, u_hat, 0.5 * u_hat**2],
        [1, u_hat + a_hat, H_hat + u_hat * a_hat]
    ])
    
    alpha = 0.5 * (gamma - 1) / a_hat**2
    l_vectors = np.array([
        [0.5 + alpha * u_hat**2, 
         - (alpha * gamma * u_hat + 0.5 / a_hat), 
         alpha * gamma - 1],
        [1 - alpha * u_hat**2, 
         gamma * alpha * u_hat, 
         - gamma * alpha],
        [0.5 + alpha * u_hat**2, 
         - (alpha * u_hat / gamma - 0.5 / a_hat), 
         alpha - 0.5 / a_hat**2]
    ])
    
    return lambda_, r_vectors, l_vectors

def delta_U(UL,UR):
    return UR-UL

def additional_flux(UL,UR):
    roe_avg_state_val = roe_avg_state(UL,UR)
    rho=roe_avg_state_val['rho']
    a=roe_avg_state_val['a']
    eigenvalues, right_eigenvectors, left_eigenvectors = eigenvalues_and_vectors(roe_avg_state_val)
    UL_original=conservative_to_primitive(UL)
    UR_original=conservative_to_primitive(UR)
    delta_U_original=UR_original-UL_original
    delta_rho=delta_U_original[0]
    delta_u=delta_U_original[1]
    delta_p=delta_U_original[2]   
    
    beta=np.zeros(3)
    beta[0]=0.5*(delta_p-rho*a*delta_u)/(a**2)
    beta[1]=delta_rho-delta_p/(a**2)
    beta[2]=0.5*(delta_p+rho*a*delta_u)/(a**2)
    
    additional_flux2 =beta[0]*right_eigenvectors[0]*np.abs(eigenvalues[0])+beta[1]*right_eigenvectors[1]*np.abs(eigenvalues[1])+beta[2]*right_eigenvectors[2]*np.abs(eigenvalues[2])
    

    return additional_flux2

def minmod(a,b):
    results = np.zeros_like(a)
    for i in range(len(a)):
        if a[i]*b[i] > 0:
            results[i] = np.sign(a[i])*min(abs(a[i]),abs(b[i]))
        else:
            results[i] = 0
    return results

    
def TVD_flux(Uj_left,Uj, Uj_right,Uj_right_right,dx,dt):
    Dj= minmod((Uj-Uj_left)/dx,(Uj_right-Uj)/dx)
    Dj_right = minmod((Uj_right-Uj)/dx,(Uj_right_right-Uj_right)/dx)
    U_jL = Uj + 0.5*Dj*dx
    U_jR = Uj_right - 0.5*Dj_right*dx
    F = np.zeros(3)
    F = 0.5*(flux(U_jL)+flux(U_jR)) - 0.5*additional_flux(U_jL,U_jR)
    return F
    
def delta_Qj(U,dx,dt):
    N = len(U)
    Q = np.zeros_like(U)
    for j in range(3,N-3):
        Q[j] = (TVD_flux(U[j-2],U[j-1],U[j],U[j+1],dx,dt) - TVD_flux(U[j-1],U[j],U[j+1],U[j+2],dx,dt))/dx
    return Q
    
       
def run_TVD(nx, t_final):
    dt = 0.001  
    x = np.linspace(0, 1, nx)
    dx = x[1] - x[0]
    U = np.array([primitive_to_conservative(initial_state(xi)) for xi in x])
    t = 0
    while t < t_final:
        U_1 = U + dt * delta_Qj(U,dx,dt)
        U_2 = 0.75*U + 0.25*(U_1 + dt*delta_Qj(U_1,dx,dt))
        U = 1/3*U + 2/3*(U_2 + dt*delta_Qj(U_2,dx,dt))
        t += dt
        
    return x, np.array([conservative_to_primitive(Ui) for Ui in U])
    

nx = 201
t_final = 0.2
x, results = run_TVD(nx, t_final)
nx = nx - 1

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(x, results[:, 0])
plt.title('Density')
plt.xlabel('x')
plt.ylabel('Density')
plt.subplot(1, 3, 2)
plt.plot(x, results[:, 1])
plt.title('Velocity')
plt.xlabel('x')
plt.ylabel('Velocity')
plt.subplot(1, 3, 3)
plt.plot(x, results[:, 2])
plt.title('Pressure')
plt.xlabel('x')
plt.ylabel('Pressure')
plt.tight_layout()
plt.show()