import matplotlib.pyplot as plt
import streamlit as st

from core.equations import rk4, sistema

st.title("Simulador de Circuito RLC com Método de Runge-Kutta")

st.image("media/circuito.png", caption="Diagrama do Circuito RLC Série", use_container_width=False)

st.markdown("### Parâmetros do Circuito")

col1, col2, col3 = st.columns(3)
with col1:
    R = st.number_input("Resistência (Ω)", min_value=0.0, max_value=100.0, value=10.0, format="%.2f")
with col2:
    L = st.number_input("Indutância (H)", min_value=0.001, max_value=1.0, value=0.1, format="%.4f")
with col3:
    C = st.number_input("Capacitância (F)", min_value=1e-6, max_value=0.1, value=0.001, format="%.6f", step=1e-6)

tipo_fonte = st.selectbox("Tipo de Fonte", ["DC", "AC"])

if tipo_fonte == "DC":
    V0 = st.number_input("Tensão Contínua (V)", min_value=0.0, max_value=50.0, value=10.0, format="%.2f")
    A = 0
    f_onda = 0
else:
    col4, col5 = st.columns(2)
    with col4:
        A = st.number_input("Amplitude da Fonte AC (V)", min_value=0.0, max_value=50.0, value=10.0, format="%.2f")
    with col5:
        f_onda = st.number_input("Frequência (Hz)", min_value=1.0, max_value=1000.0, value=50.0, format="%.2f")
    V0 = 0

tf = st.number_input("Tempo total de simulação (s)", min_value=0.01, max_value=10.0, value=1.0, format="%.4f")
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
