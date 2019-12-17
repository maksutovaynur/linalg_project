import wave

import numpy as np

def derivG2(u):
    return u*np.exp(-(1/2)*u**2)

def deriv_deriv_G(u):
    return (1 - u**2)*np.exp(-(1/2)*u**2)

def converge_criterium(prev_w, curr_w, eps=0.01):
    if np.abs(prev_w.T@curr_w) < (1 - eps):
        return True
    else:
        return False

def fastICA(mixed, eps):
    curr_w = np.random.rand(mixed.shape[0])
    prev_w = np.random.rand(mixed.shape[0])
    curr_w = curr_w / np.linalg.norm(curr_w)
    prev_w = prev_w / np.linalg.norm(prev_w)
    while converge_criterium(prev_w, curr_w, eps):
        prev_w = curr_w.copy()
        curr_w = (((mixed @ derivG2(curr_w.T @ mixed)) / mixed.shape[1]) 
                  - curr_w*(deriv_deriv_G(curr_w.T @ mixed)).mean())
        curr_w = curr_w / np.linalg.norm(curr_w)
    return curr_w