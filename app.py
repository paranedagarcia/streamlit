import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ---------------------------------------------------------
# CONFIGURACIÓN BÁSICA
# ---------------------------------------------------------
st.set_page_config(
    page_title="Análisis Bayes Internet - Soria",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Análisis Bayesiano de Cortes de Internet en Soria")
st.markdown(
"""
Este aplicativo interactivo permite:

- Ajustar probabilidades de operadores y routers  
- Calcular la probabilidad de corte total  
- Aplicar el **Teorema de Bayes**  
- Simular usuarios con **Monte Carlo**  
- Visualizar el modelo probabilístico

> Modelo simplificado con fines pedagógicos.
"""
)

# ---------------------------------------------------------
# SIDEBAR: PARÁMETROS DEL MODELO
# ---------------------------------------------------------
st.sidebar.header("⚙️ Parámetros del modelo")

st.sidebar.subheader("Operadores")
P_T = st.sidebar.slider("P(Timofónica)", 0.0, 1.0, 0.60, 0.01)
P_R = 1 - P_T
st.sidebar.write(f"P(Robafone) = 1 - P(T) = **{P_R:.2f}**")

st.sidebar.subheader("Routers")
P_X = st.sidebar.slider("P(router Xisco)", 0.0, 1.0, 0.70, 0.01)
P_N = 1 - P_X
st.sidebar.write(f"P(router Nuaweii) = 1 - P(X) = **{P_N:.2f}**")

st.sidebar.subheader("Probabilidades de corte (condicionadas)")
P_C_T = st.sidebar.slider("P(corte | Timofónica)", 0.0, 0.5, 0.10, 0.01)
P_C_R = st.sidebar.slider("P(corte | Robafone)", 0.0, 0.5, 0.15, 0.01)
P_C_X = st.sidebar.slider("P(corte | router Xisco)", 0.0, 0.5, 0.05, 0.01)

# Asunción extra para simulación: prob corte con Nuaweii
st.sidebar.markdown("---")
st.sidebar.subheader("Suposición adicional (para simulación)")
P_C_N = st.sidebar.slider(
    "P(corte | router Nuaweii) (asumida)", 0.0, 0.5, 0.20, 0.01,
    help="No viene del enunciado original: se asume con fines de simulación."
)

N_sim = st.sidebar.number_input(
    "N simulaciones Monte Carlo",
    min_value=10_000,
    max_value=2_000_000,
    value=200_000,
    step=10_000
)

# ---------------------------------------------------------
# CÁLCULOS TEÓRICOS
# ---------------------------------------------------------
st.header("1️⃣ Cálculos teóricos")

st.markdown("### 1.1 Probabilidad total de corte")

P_C = P_C_T * P_T + P_C_R * P_R

col1, col2, col3 = st.columns(3)
col1.metric("P(C) corte total", f"{P_C:.4f}", help="Probabilidad promedio de corte en la población")

st.markdown(
f"""
Usando la **probabilidad total**:

\\[
P(C) = P(C\\mid T)P(T) + P(C\\mid R)P(R) = {P_C_T:.2f}\\cdot{P_T:.2f} + {P_C_R:.2f}\\cdot{P_R:.2f} = {P_C:.4f}
\\]
"""
)

st.markdown("### 1.2 Probabilidad de tener Xisco si el usuario tiene corte, P(X | C)")

if P_C > 0:
    P_X_C = P_C_X * P_X / P_C
else:
    P_X_C = np.nan

col1, col2, col3 = st.columns(3)
col2.metric("P(X | C)", f"{P_X_C:.4f}" if not np.isnan(P_X_C) else "N/A",
            help="Probabilidad de que el usuario use router Xisco dado que tiene un corte")

st.markdown(
f"""
Aplicando el **Teorema de Bayes**:

\\[
P(X\\mid C) = \\frac{{P(C\\mid X)P(X)}}{{P(C)}} 
= \\frac{{{P_C_X:.2f}\\cdot{P_X:.2f}}}{{{P_C:.4f}}}
= {P_X_C:.4f}
\\]

Pedagógicamente, esto responde:

> *Dado que un usuario tiene la línea cortada, ¿qué tan probable es que tenga un router Xisco en casa?*
"""
)

st.markdown("### 1.3 Probabilidad de corte si NO tiene router Xisco, P(C | ¬X)")

if P_N > 0:
    P_C_and_X = P_C_X * P_X
    P_C_not_X = (P_C - P_C_and_X) / P_N
else:
    P_C_not_X = np.nan

col1, col2, col3 = st.columns(3)
col3.metric("P(C | ¬X)", f"{P_C_not_X:.4f}" if not np.isnan(P_C_not_X) else "N/A",
            help="Probabilidad de corte en usuarios que NO utilizan router Xisco")

st.markdown(
f"""
Utilizamos la siguiente relación:

\\[
P(C\\mid \\neg X) = \\frac{{P(C) - P(C \\cap X)}}{{P(\\neg X)}}
\\]

donde:

- \\(P(C \\cap X) = P(C\\mid X)P(X) = {P_C_X:.2f}\\cdot{P_X:.2f} = {P_C_and_X:.4f}\\)  
- \\(P(\\neg X) = 1 - P(X) = {P_N:.2f}\\)

Sustituyendo:

\\[
P(C\\mid \\neg X) = \\frac{{{P_C:.4f} - {P_C_and_X:.4f}}}{{{P_N:.2f}}} = {P_C_not_X:.4f}
\\]

Interpretación:

> El riesgo de corte entre usuarios que **no** usan Xisco es bastante mayor que el riesgo asociado exclusivamente a Xisco (P(C|X)).
"""
)

# ---------------------------------------------------------
# VISUALIZACIONES
# ---------------------------------------------------------
st.header("2️⃣ Visualizaciones")

tab1, tab2 = st.tabs(["📊 Comparación de probabilidades", "📈 Grafo probabilístico"])

with tab1:
    st.subheader("Comparación de probabilidades clave")

    labels = ["P(C)", "P(X|C)", "P(C|¬X)", "P(C|X)"]
    values = [P_C,
              P_X_C if not np.isnan(P_X_C) else 0,
              P_C_not_X if not np.isnan(P_C_not_X) else 0,
              P_C_X]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
    ax.set_ylabel("Probabilidad")
    ax.set_title("Probabilidades teóricas comparadas")
    for i, v in enumerate(values):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center")
    st.pyplot(fig)

    st.markdown(
    """
    - **P(C)**: riesgo promedio de corte en la población  
    - **P(X|C)**: fracción de usuarios con Xisco entre quienes sufren corte  
    - **P(C|¬X)**: riesgo de corte entre quienes **no** usan Xisco  
    - **P(C|X)**: riesgo de corte exclusivamente entre usuarios con router Xisco  
    """
    )

with tab2:
    st.subheader("Diagrama causal (grafo probabilístico)")

    G = nx.DiGraph()
    G.add_edges_from([("Operador", "Corte"), ("Router", "Corte")])

    pos = {
        "Operador": (0, 1),
        "Router": (2, 1),
        "Corte": (1, 0)
    }

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    nx.draw(G, pos, with_labels=True, node_size=3000, font_size=12, ax=ax2)
    ax2.set_title("Diagrama probabilístico del problema")
    st.pyplot(fig2)

    st.markdown(
    """
    En este grafo:

    - El **Operador** influye en el evento *Corte* mediante \\(P(C\\mid T), P(C\\mid R)\\)  
    - El **Router** influye en el evento *Corte* mediante \\(P(C\\mid X), P(C\\mid N)\\)  
    - El nodo **Corte** resume el resultado final del sistema  
    """
    )

# ---------------------------------------------------------
# SIMULACIÓN MONTE CARLO
# ---------------------------------------------------------
st.header("3️⃣ Simulación Monte Carlo")

st.markdown(
"""
En esta sección generamos usuarios simulados y analizamos:

- La frecuencia empírica de corte  
- La proporción de Xisco entre los usuarios con corte  
- El riesgo de corte entre usuarios sin Xisco  

> ⚠️ Nota: para simular necesitamos una **suposición adicional**:  
> usamos también \\(P(C\\mid N)\\), ajustable en la barra lateral.
"""
)

if st.button("▶️ Ejecutar simulación Monte Carlo"):
    # Simulación operador
    operators = np.random.choice(["T", "R"], size=N_sim, p=[P_T, P_R])
    # Simulación router
    routers = np.random.choice(["X", "N"], size=N_sim, p=[P_X, P_N])

    # Cortes por operador
    rand_op = np.random.rand(N_sim)
    cuts_op = np.where(
        operators == "T",
        rand_op < P_C_T,
        rand_op < P_C_R
    )

    # Cortes por router
    rand_rt = np.random.rand(N_sim)
    cuts_rt = np.where(
        routers == "X",
        rand_rt < P_C_X,
        rand_rt < P_C_N
    )

    # Modelo OR: hay corte si alguno de los mecanismos provoca corte
    cuts = cuts_op | cuts_rt

    # Estimaciones empíricas
    MC_P_C = cuts.mean()
    MC_P_X_C = (routers[cuts] == "X").mean() if cuts.any() else np.nan
    MC_P_C_not_X = cuts[routers == "N"].mean() if (routers == "N").any() else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("MC: P(C)", f"{MC_P_C:.4f}", delta=f"{MC_P_C - P_C:+.4f}")
    c2.metric("MC: P(X | C)", f"{MC_P_X_C:.4f}", delta=f"{MC_P_X_C - P_X_C:+.4f}" if not np.isnan(MC_P_X_C) else "N/A")
    c3.metric("MC: P(C | ¬X)", f"{MC_P_C_not_X:.4f}", delta=f"{MC_P_C_not_X - P_C_not_X:+.4f}" if not np.isnan(MC_P_C_not_X) else "N/A")

    st.markdown(
    """
    - Los **valores teóricos** provienen de las fórmulas de Bayes y probabilidad total.  
    - Los **valores Monte Carlo** provienen de simular muchos usuarios y estimar las frecuencias.  

    Cuanto mayor es el número de simulaciones (N), más se aproximan los resultados de Monte Carlo a los teóricos,
    siempre que el modelo de simulación sea coherente con las hipótesis teóricas.
    """
    )
else:
    st.info("Haz clic en **▶️ Ejecutar simulación Monte Carlo** para generar los resultados empíricos.")

# ---------------------------------------------------------
# INTERPRETACIÓN FINAL
# ---------------------------------------------------------
st.header("4️⃣ Interpretación pedagógica")

st.markdown(
f"""
- **Riesgo global de corte (P(C))**: alrededor de **{P_C:.2%}**, determinado por la mezcla de operadores.  
- **P(X | C)**: mide qué tan frecuentes son los usuarios con router Xisco **entre quienes sufren un corte**.  
- **P(C | ¬X)**: cuantifica el riesgo de corte si el usuario **no** usa Xisco; contrastarlo con P(C|X) permite evaluar el impacto del router.  

Este tipo de análisis es típico en:

- Evaluación de **riesgo operacional** (operadoras, ISPs)  
- Análisis de **fiabilidad de hardware** (routers, ONTs, etc.)  
- Estudios de **calidad de servicio (QoS)** en telecomunicaciones  

La combinación de **modelo teórico** + **simulación Monte Carlo** refuerza la comprensión:

- El modelo teórico aporta fórmulas cerradas e interpretables.  
- Monte Carlo aporta una vista *empírica* y conectada con datos reales.  
"""
)
