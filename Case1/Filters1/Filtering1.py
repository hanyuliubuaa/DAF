import numpy as np
from EKF import EKF
from UKF import UKF
from PF import PF
from CRLB import CRLB
from J_h import J_h
from f_EKF import f_EKF
from h_EKF import h_EKF
from f import f
from h import h
import time
import matplotlib.pyplot as plt
np.random.seed(1)

X_TRUE = np.loadtxt("data1/Test_true_filter.txt", dtype='float64')
Z_TRUE = np.loadtxt("data1/Test_data.txt", dtype='float64')
Q = np.diag([1e-4, 1e-4, 1e-4])
R = np.diag([100., 100., 100.])
T = 0.03
N = 210
M = 10
nz = 3
L_warm = 10
err_UKF = np.zeros([3, N, M])
err_EKF = np.zeros([3, N, M])
err_PF_1 = np.zeros([3, N, M])
err_PF_2 = np.zeros([3, N, M])
err_PCRB = np.zeros([3, 3, N, M])
t_UKF = 0.
t_EKF = 0.
t_PF_1 = 0.
t_PF_2 = 0.
t_PCRB = 0.
t_initial = 0.

def non_equ(x, z):
    Z = []
    for i in range(L_warm):
        if i == 0:
            Z.append((h(x) - z[:, i]) / 10)
        else:
            x_pred = f(x, [0, T * i])
            z_pred = h(x_pred)
            Z.append((z_pred - z[:, i]) / 10)
    Z = np.hstack(Z)
    return Z

def jac_non_equ(x, z):
    H_total = []
    for i in range(L_warm):
        if i == 0:
            H_total.append(J_h(x) / 10)
        else:
            x_pred, Phi = f_EKF(x, [0, T * i])
            H = J_h(x_pred)
            H_total.append((H / 10) @ Phi)
    H_total = np.vstack(H_total)
    return H_total

for times in range(M):
    print(times)
    x_true = np.reshape(X_TRUE[times, :], (3, N))
    z = np.reshape(Z_TRUE[times, :], (nz, N))
    start_time = time.time()
    x = x_true[:, 0]
    for count in range(1000):
        H_total = []
        Z = []
        R_total = np.zeros((nz * L_warm, nz * L_warm))
        for i in range(L_warm):
            if i == 0:
                Z.append(z[:, i] - h(x))
                H_total.append(J_h(x))
                R_total[nz*i:nz*(i+1), nz*i:nz*(i+1)] = R
            else:
                x_pred, Phi = f_EKF(x, [0, T * i])
                z_pred, H = h_EKF(x_pred)
                Z.append(z[:, i] - z_pred)
                H_total.append(H @ Phi)
                R_total[nz*i:nz*(i+1), nz*i:nz*(i+1)] = R
        H_total = np.vstack(H_total)
        Z = np.hstack(Z)
        H_total_T = H_total.T
        delta = np.linalg.solve(H_total_T @ np.linalg.inv(R_total) @ H_total, H_total_T @ np.linalg.inv(R_total) @ Z)
        if np.linalg.norm(delta) < 1e-3:
            break
        if count <= 19:
            x += delta
        else:
            x += delta / 10
    x = x_pred
    P = Phi @ np.linalg.inv(H_total_T @ np.linalg.inv(R_total) @ H_total) @ Phi.T
    if np.linalg.norm(delta) >= 1e-3:
        raise ValueError("Least square fails!")
    
    P = 0.5*(P + P.T)
    t_initial = t_initial + time.time() - start_time

    x_UKF = np.zeros([3, N])
    P_UKF = np.zeros([3, 3, N])
    x_UKF[:, L_warm-1] = x
    P_UKF[:, :, L_warm-1] = P
    x_EKF = np.copy(x_UKF)
    P_EKF = np.copy(P_UKF)
    
    N_particles_1 = 100
    w_1 = np.ones(N_particles_1) / N_particles_1
    particles_1 = np.zeros((3, N_particles_1))
    for i in range(N_particles_1):
        particles_1[:, i] = np.random.multivariate_normal(x, P)
    x_PF_1 = np.copy(x_UKF)
    
    N_particles_2 = 500
    w_2 = np.ones(N_particles_2) / N_particles_2
    particles_2 = np.zeros((3, N_particles_2))
    for i in range(N_particles_2):
        particles_2[:, i] = np.random.multivariate_normal(x, P)
    x_PF_2 = np.copy(x_UKF)
    
    J = np.zeros((3, 3, N))
    J[:, :, 9] = np.linalg.inv(P)
    PCRB = np.copy(P_UKF)

    start_time = time.time()
    for k in range(L_warm, N):
        x_UKF[:, k], P_UKF[:, :, k] = UKF(x_UKF[:, k-1], P_UKF[:, :, k-1], z, Q, R, k, T)
    t_UKF = t_UKF + time.time() - start_time
    
    start_time = time.time()
    for k in range(L_warm, N):
        x_EKF[:, k], P_EKF[:, :, k] = EKF(x_EKF[:, k-1], P_EKF[:, :, k-1], z, Q, R, k, T)
    t_EKF = t_EKF + time.time() - start_time
    
    start_time = time.time()
    for k in range(L_warm, N):
        [particles_1, w_1, x_PF_1[:, k]] = PF(particles_1, w_1, z, Q, R, k, T)
    t_PF_1 = t_PF_1 + time.time() - start_time
    
    start_time = time.time()
    for k in range(L_warm, N):
        [particles_2, w_2, x_PF_2[:, k]] = PF(particles_2, w_2, z, Q, R, k, T)
    t_PF_2 = t_PF_2 + time.time() - start_time
    
    start_time = time.time()
    for k in range(L_warm, N):
        J[:, :, k], PCRB[:, :, k] = CRLB(J[:, :, k-1], x_true[:, k-1], x_true[:, k], Q, R, T)
    t_PCRB = t_PCRB + time.time() - start_time
    
    err_UKF[:, :, times] = x_UKF - x_true
    err_EKF[:, :, times] = x_EKF - x_true
    err_PF_1[:, :, times] = x_PF_1 - x_true
    err_PF_2[:, :, times] = x_PF_2 - x_true
    err_PCRB[:, :, :, times] = PCRB
    
    
RMSE_UKF = np.sqrt(np.mean(np.mean(err_UKF[:, L_warm:, :] ** 2, axis=2), axis=0))
RMSE_EKF = np.sqrt(np.mean(np.mean(err_EKF[:, L_warm:, :] ** 2, axis=2), axis=0))
RMSE_PF_1 = np.sqrt(np.mean(np.mean(err_PF_1[:, L_warm:, :] ** 2, axis=2), axis=0))
RMSE_PF_2 = np.sqrt(np.mean(np.mean(err_PF_2[:, L_warm:, :] ** 2, axis=2), axis=0))
MSE_PCRB = np.mean(err_PCRB[:, :, L_warm:, :], axis=3)
RMSE_PCRB = np.sqrt(
    (np.squeeze(MSE_PCRB[0, 0, :]) +
     np.squeeze(MSE_PCRB[1, 1, :]) +
     np.squeeze(MSE_PCRB[2, 2, :])) / 3
)
norm_RMSE_UKF = np.sqrt(np.mean(RMSE_UKF ** 2))
norm_RMSE_EKF = np.sqrt(np.mean(RMSE_EKF ** 2))
norm_RMSE_PF_1 = np.sqrt(np.mean(RMSE_PF_1 ** 2))
norm_RMSE_PF_2 = np.sqrt(np.mean(RMSE_PF_2 ** 2))
norm_RMSE_PCRB = np.sqrt(np.mean(RMSE_PCRB ** 2))


print('norm_RMSE_UKF', norm_RMSE_UKF)
print('norm_RMSE_EKF', norm_RMSE_EKF)
print('norm_RMSE_PF_1', norm_RMSE_PF_1)
print('norm_RMSE_PF_2', norm_RMSE_PF_2)
print('norm_RMSE_CLRB', norm_RMSE_PCRB)
print('t_initial', t_initial)
print('t_UKF', t_UKF)
print('t_EKF', t_EKF)
print('t_PF_1', t_PF_1)
print('t_PF_2', t_PF_2)
print('t_CLRB', t_PCRB)
# np.savetxt('RMSE_UKF_1.txt', RMSE_UKF)
# np.savetxt('RMSE_EKF_1.txt', RMSE_EKF)
# np.savetxt('RMSE_PF_1_1.txt', RMSE_PF_1)
# np.savetxt('RMSE_PF_2_1.txt', RMSE_PF_2)
# np.savetxt('RMSE_CLRB_1.txt', RMSE_PCRB)

# plt.plot(np.arange(0, 200, 1), RMSE_EKF, 'k')
# plt.plot(np.arange(0, 200, 1), RMSE_UKF, 'r')
# plt.plot(np.arange(0, 200, 1), RMSE_PF_1, 'g')
# plt.plot(np.arange(0, 200, 1), RMSE_PF_2, 'm')
# plt.plot(np.arange(0, 200, 1), RMSE_PCRB, 'c')
# plt.legend(['EKF', 'UKF', 'PF(100)', 'PF(500)', 'CLRB'])
# plt.show()
# plt.savefig('Error_1.png')