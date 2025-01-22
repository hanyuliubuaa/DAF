import numpy as np
from scipy.stats import multivariate_normal
from f import f
from h import h

def GaussPDF(x, M, P):
    # To ensure exp(.) not be zero or inf.
    n = M.shape[0]
    N = M.shape[1]
    PDF = np.zeros(N)
    
    for i in range(N):
        PDF[i] = -0.5 * (x - M[:, i]).T @ np.linalg.inv(P) @ (x - M[:, i])
    
    PDF_1, i = np.min(PDF), np.argmin(PDF)
    PDF_2, j = np.max(PDF), np.argmax(PDF)
    flag = len(np.where(PDF == PDF_2)[0])
    delta = PDF_2 - PDF_1
    
    if delta == 0:
        PDF = np.ones(N) / N
    else:
        while delta > 400:
            PDF[i] = PDF_2
            PDF_1, i = np.min(PDF), np.argmin(PDF)
            delta = PDF_2 - PDF_1
        
        if delta == 0 and flag == 1:
            PDF = np.zeros(N)
            PDF[j] = 1
        else:
            detP1 = np.linalg.det(P)
            indexP1 = -0.5 * (x - M[:, i]).T @ np.linalg.inv(P) @ (x - M[:, i])
            PDF_1 = 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(detP1)) * np.exp(indexP1)
            alpha = 0
            
            while PDF_1 < 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(detP1)) * np.exp(-delta):
                alpha += 1
                PDF_1 = 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(detP1)) * np.exp(indexP1 + alpha)
            
            for i in range(N):
                PDF[i] = 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(np.linalg.det(P))) * \
                         np.exp(-0.5 * (x - M[:, i]).T @ np.linalg.inv(P) @ (x - M[:, i]) + alpha)

    return PDF

def PF(particles, w, z, Q, R, k, T):
    tspan = [0, T]
    n = particles.shape[0]
    N = particles.shape[1]  # Number of particles

    # Propagate particles
    particles = f(particles, tspan)
    particles += np.random.multivariate_normal([0, 0, 0], Q, N).T

    # Update weights
    PDF = GaussPDF(z[:, k], h(particles), R)
    w *= PDF
    w /= np.sum(w)  # normalization

    # Resampling
    if 1 / np.sum(w ** 2) < N / 2:
        cdf = np.cumsum(w)
        positions = np.random.rand(N)
        indexes = np.searchsorted(cdf, positions)
        particles = particles[:, indexes]
        w = np.ones(N) / N

    # Calculating the mean
    x_esti = np.dot(particles, w)

    return particles, w, x_esti