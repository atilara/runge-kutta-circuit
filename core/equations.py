import numpy as np


def fonte_v(t, tipo, V0=0, A=0, f=0):
    if tipo == "DC":
        return V0
    elif tipo == "AC":
        return A * np.sin(2 * np.pi * f * t)
    return 0

def sistema(t, x, R, L, C, tipo_fonte, V0=0, A=0, f=0):
    x1, x2 = x
    Vt = fonte_v(t, tipo_fonte, V0, A, f)
    dx1dt = x2
    dx2dt = (1 / L) * (Vt - R * x2 - x1 / C)
    return np.array([dx1dt, dx2dt])

def rk4(f, x0, t0, tf, h, R, L, C, tipo_fonte, V0=0, A=0, f_onda=0.0):
    t_vals = np.arange(t0, tf, h)
    x_vals = np.zeros((len(t_vals), len(x0)))
    x_vals[0] = x0
    for i in range(1, len(t_vals)):
        t = t_vals[i - 1]
        x = x_vals[i - 1]
        k1 = h * f(t, x, R, L, C, tipo_fonte, V0, A, f_onda)
        k2 = h * f(t + h / 2, x + k1 / 2, R, L, C, tipo_fonte, V0, A, f_onda)
        k3 = h * f(t + h / 2, x + k2 / 2, R, L, C, tipo_fonte, V0, A, f_onda)
        k4 = h * f(t + h, x + k3, R, L, C, tipo_fonte, V0, A, f_onda)
        x_vals[i] = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t_vals, x_vals