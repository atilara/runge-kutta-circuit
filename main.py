import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


# === Função da fonte V(t) ===
def fonte_v(t, tipo, V0=0, A=0, f=0):
    if tipo == "DC":
        return V0
    elif tipo == "AC":
        return A * np.sin(2 * np.pi * f * t)
    return 0


# === Sistema de EDOs ===
def sistema(t, x, R, L, C, tipo_fonte, V0=0, A=0, f=0):
    x1, x2 = x
    Vt = fonte_v(t, tipo_fonte, V0, A, f)
    dx1dt = x2
    dx2dt = (1 / L) * (Vt - R * x2 - x1 / C)
    return np.array([dx1dt, dx2dt])


# === Runge-Kutta 4ª ordem ===
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


# === Interface Streamlit ===
st.title("Simulador de Circuito RLC com Método de Runge-Kutta")

# Escolha dos parâmetros
st.sidebar.header("Parâmetros do Circuito")

R = st.sidebar.slider("Resistência (Ω)", min_value=0.0, max_value=100.0, value=10.0)
L = st.sidebar.slider("Indutância (H)", min_value=0.001, max_value=1.0, value=0.1)
C = st.sidebar.slider(
    "Capacitância (F)", min_value=0.00001, max_value=0.01, value=0.001
)

tipo_fonte = st.sidebar.selectbox("Tipo de Fonte", ["DC", "AC"])

if tipo_fonte == "DC":
    V0 = st.sidebar.slider(
        "Tensão Contínua (V)", min_value=0.0, max_value=50.0, value=10.0
    )
    A = 0
    f_onda = 0
else:
    A = st.sidebar.slider(
        "Amplitude da Fonte AC (V)", min_value=0.0, max_value=50.0, value=10.0
    )
    f_onda = st.sidebar.slider(
        "Frequência (Hz)", min_value=1.0, max_value=1000.0, value=50.0
    )
    V0 = 0

# Tempo de simulação
tf = st.sidebar.slider("Tempo total de simulação (s)", 0.01, 1.0, 0.1)
h = 0.0001

# Executar simulação
x0 = [0, 0]  # Carga e Corrente iniciais
t_vals, x_vals = rk4(
    sistema, x0, 0, tf, h, R, L, C, tipo_fonte, V0=V0, A=A, f_onda=f_onda  # type: ignore
)

# Plot
st.subheader("Corrente vs Tempo")
fig, ax = plt.subplots()
ax.plot(t_vals, x_vals[:, 1], label="Corrente (A)", color="blue")
ax.set_xlabel("Tempo (s)")
ax.set_ylabel("Corrente (A)")
ax.grid(True)
ax.legend()
st.pyplot(fig)
