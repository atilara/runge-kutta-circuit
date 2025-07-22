import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

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

st.title("Simulador de Circuito RLC com Método de Runge-Kutta")

st.sidebar.header("Parâmetros do Circuito")

R = st.sidebar.number_input("Resistência (Ω)", min_value=0.0, max_value=100.0, value=10.0, format="%.2f")
L = st.sidebar.number_input("Indutância (H)", min_value=0.001, max_value=1.0, value=0.1, format="%.4f")
C = st.sidebar.number_input(
    "Capacitância (F)",
    min_value=1e-6,
    max_value=0.1,
    value=0.001,
    format="%.6f",
    step=1e-6,
)

tipo_fonte = st.sidebar.selectbox("Tipo de Fonte", ["DC", "AC"])

if tipo_fonte == "DC":
    V0 = st.sidebar.number_input("Tensão Contínua (V)", min_value=0.0, max_value=50.0, value=10.0, format="%.2f")
    A = 0
    f_onda = 0
else:
    A = st.sidebar.number_input("Amplitude da Fonte AC (V)", min_value=0.0, max_value=50.0, value=10.0, format="%.2f")
    f_onda = st.sidebar.number_input("Frequência (Hz)", min_value=1.0, max_value=1000.0, value=50.0, format="%.2f")
    V0 = 0

tf = st.sidebar.number_input("Tempo total de simulação (s)", min_value=0.01, max_value=1.0, value=0.1, format="%.4f")
h = 0.0001

x0 = [0, 0]
t_vals, x_vals = rk4(sistema, x0, 0, tf, h, R, L, C, tipo_fonte, V0=V0, A=A, f_onda=f_onda)

st.markdown("""
    <style>
    body { background-color: #f0f4f8; }
    </style>
""", unsafe_allow_html=True)

if tipo_fonte == "DC":
    fonte_str = f"{V0:.2f}"
else:
    fonte_str = rf"{A:.2f} \sin(2 \pi {f_onda:.2f} t)"

formula_latex = rf"""
L \frac{{d^2q}}{{dt^2}} + R \frac{{dq}}{{dt}} + \frac{{q}}{{C}} = {fonte_str}
\\[10pt]
L = {L:.4f} \text{{ H}}, \quad R = {R:.2f} \, \Omega, \quad C = {C:.6f} \text{{ F}}
"""

st.markdown("### Equação do Circuito RLC Série com parâmetros atuais:")
st.latex(formula_latex)

st.subheader("Corrente vs Tempo")
fig, ax = plt.subplots()
ax.plot(t_vals, x_vals[:, 1], label="Corrente (A)", color="#e63946", linewidth=2.0)
ax.set_facecolor("#f1faee")
fig.patch.set_facecolor("#f1faee")
ax.tick_params(colors="#1d3557")
ax.spines["bottom"].set_color("#1d3557")
ax.spines["left"].set_color("#1d3557")
ax.xaxis.label.set_color("#1d3557")
ax.yaxis.label.set_color("#1d3557")
ax.set_xlabel("Tempo (s)")
ax.set_ylabel("Corrente (A)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
st.pyplot(fig)
