import numpy as np
import quaternion # https://quaternion.readthedocs.io/en/latest/

def qssim(img1, img2):

    q1, q2 = img1.astype(np.quaternion), img2.astype(np.quaternion)
    
    # média e desvio padrão dos quaterniônicos
    uq1, uq2 = np.mean(q1), np.mean(q2)    
    sigma1, sigma2 = np.std(q1), np.std(q2)
    
    # covariância
    cov = np.mean((q1 - uq1) * (np.conj(q2 - uq2)))
    
    qssim_score = 2 * ((2 * uq1 * uq2) / (uq1 ** 2 + uq2 ** 2)) * (cov / (sigma1 ** 2 + sigma2 ** 2))
    return np.abs(qssim_score)