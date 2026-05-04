"""
Impurity Transport Lab — EAST
Simulador interactivo de transporte de impurezas en tokamaks.

Autores: Vicente Sepúlveda · PUC · Tesis EAST 2026
Stack:   Streamlit + Plotly + Aurora (opcional)
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import transport as tr
import aurora_runner as ar

# ── Configuración ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Impurity Transport Lab",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

RHO = tr.RHO

# ── Estilos ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
  .metric-box { background: #f8f9fa; border-radius: 8px; padding: 12px 16px;
                text-align: center; border: 0.5px solid #e0e0e0; }
  .metric-label { font-size: 12px; color: #666; margin-bottom: 2px; }
  .metric-value { font-size: 22px; font-weight: 600; color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚛ Impurity Transport Lab — EAST")
st.caption("Simulador de transporte radial de impurezas en geometría cilíndrica 1D")

aurora_ok = ar.aurora_status()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuración")

    # Modo de física
    st.subheader("Modelo")
    use_aurora = st.toggle(
        "Usar Aurora (física atómica real)",
        value=False,
        disabled=not aurora_ok,
        help="Requiere `aurora-fusion` instalado. Usa datos ADAS reales para ionización y PECs."
    )
    if not aurora_ok:
        st.caption("⚠ Aurora no instalado — modo analítico activo")

    species = st.selectbox("Impureza", ["W", "Fe", "Mo"], index=0)

    st.divider()

    # Perfiles de plasma
    st.subheader("Plasma (EAST)")
    Te0  = st.slider("T_e(0) [keV]",  0.5, 8.0,  3.0, 0.1)
    ne0  = st.slider("n_e(0) [×10¹⁹ m⁻³]", 0.5, 10.0, 3.0, 0.1)
    ne0_si = ne0 * 1e19

    st.divider()

    # Modo de transporte
    st.subheader("Transporte")
    transport_mode = st.radio(
        "Modo D / v",
        ["Uniforme", "Perfil D(ρ)/v(ρ)"],
        horizontal=True
    )

    if transport_mode == "Uniforme":
        D_val = st.select_slider(
            "D [m²/s]",
            options=[0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0],
            value=0.5
        )
        v_val = st.slider("v [m/s]  (− = pinch inward)", -25.0, 25.0, -5.0, 0.5)
        D_prof = np.full_like(RHO, D_val)
        v_prof = np.full_like(RHO, v_val)

    else:
        st.caption("Define D y v en 6 puntos radiales (interpolación cúbica)")

        ctrl_pts = tr.CTRL_POINTS
        D_ctrl, v_ctrl = [], []

        cols = st.columns(2)
        with cols[0]:
            st.markdown("**D [m²/s]**")
            for rp in ctrl_pts:
                val = st.number_input(f"ρ={rp:.1f}", 0.01, 5.0, 0.5, 0.05,
                                      key=f"D_{rp}", label_visibility="visible")
                D_ctrl.append(val)
        with cols[1]:
            st.markdown("**v [m/s]**")
            for rp in ctrl_pts:
                val = st.number_input(f"ρ={rp:.1f}", -30.0, 30.0, -5.0, 1.0,
                                      key=f"v_{rp}", label_visibility="visible")
                v_ctrl.append(val)

        D_prof = tr.build_profile(RHO, D_ctrl)
        v_prof = tr.build_profile(RHO, v_ctrl, kind="linear")

    st.divider()
    st.caption("Vicente Sepúlveda · PUC · 2026")

# ── Cálculo de perfiles ───────────────────────────────────────────────────────
Te_arr, ne_arr = tr.plasma_profiles(RHO, Te0=Te0, ne0=ne0_si)

if use_aurora and aurora_ok:
    with st.spinner("Corriendo Aurora…"):
        try:
            res = ar.run_aurora_sim(RHO, Te_arr, ne_arr, D_prof, v_prof, species=species)
            nW_arr  = res['nz'].sum(axis=1)
            nW_arr /= nW_arr[-1]
            eps_arr = res['emiss']
            nz_mat  = res['nz']
            cs_lbls = res['cs_labels']
        except Exception as e:
            st.error(f"Error en Aurora: {e}")
            use_aurora = False

if not (use_aurora and aurora_ok):
    nW_arr = tr.solve_transport(RHO, D_prof, v_prof)
    eps_arr = tr.emissivity_profile(RHO, nW_arr, ne_arr, Te_arr, species=species)
    nz_mat, cs_lbls = None, None

peak = tr.peaking_factor(eps_arr)
vD_ratio = np.mean(v_prof / np.maximum(D_prof, 1e-6))

# ── Métricas ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
regime = ("Pinch inward" if vD_ratio < -2
          else "Outward" if vD_ratio > 2
          else "Difusión pura")
regime_color = "#185FA5" if vD_ratio < -2 else "#993C1D" if vD_ratio > 2 else "#5F5E5A"

for col, label, value in [
    (c1, "Factor de pico ε(0)/ε(a)", f"{peak:.2f}"),
    (c2, "⟨v/D⟩ [m⁻¹]",             f"{vD_ratio:.1f}"),
    (c3, "T_e central [keV]",         f"{Te_arr[0]:.1f}"),
    (c4, "Régimen",                    regime),
]:
    with col:
        color = regime_color if label == "Régimen" else "#1a1a2e"
        st.markdown(f"""
        <div class="metric-box">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="color:{color};font-size:{'16px' if label=='Régimen' else '22px'}">{value}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Perfiles radiales", "⚗ Estados de carga", "🗺 Mapa D–v"])

COLORS = {"W": "#185FA5", "Fe": "#993C1D", "Mo": "#0F6E56"}
IMP_COLOR = COLORS.get(species, "#185FA5")

# ── Tab 1: Perfiles radiales ──────────────────────────────────────────────────
with tab1:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Densidad de impureza  n_imp(ρ) / n_edge",
                                        f"Emisividad EUV  ε_{species}(ρ)"])

    # n_imp
    fig.add_trace(go.Scatter(x=RHO, y=nW_arr / nW_arr[-1], name=f"n_{species}",
                             line=dict(color=IMP_COLOR, width=3)), row=1, col=1)

    # Perfiles de fondo (referencia)
    fig.add_trace(go.Scatter(x=RHO, y=Te_arr / Te_arr[0], name="Te (norm.)",
                             line=dict(color="#E24B4A", width=1.5, dash="dash"),
                             opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=RHO, y=ne_arr / ne_arr[0], name="ne (norm.)",
                             line=dict(color="#378ADD", width=1.5, dash="dot"),
                             opacity=0.7), row=1, col=1)

    # Emisividad
    fig.add_trace(go.Scatter(x=RHO, y=eps_arr, name=f"ε_{species}",
                             line=dict(color=IMP_COLOR, width=3),
                             fill="tozeroy", fillcolor="rgba(24,95,165,0.15)",
                             row=1, col=2))

    # D y v normalizados
    D_n = D_prof / D_prof.max()
    v_n = (v_prof - v_prof.min()) / (v_prof.max() - v_prof.min() + 1e-9)
    fig.add_trace(go.Scatter(x=RHO, y=D_n, name="D (norm.)",
                             line=dict(color="#639922", width=1.5, dash="longdash"),
                             opacity=0.6), row=1, col=2)
    fig.add_trace(go.Scatter(x=RHO, y=v_n, name="v (norm.)",
                             line=dict(color="#BA7517", width=1.5, dash="dashdot"),
                             opacity=0.6), row=1, col=2)

    fig.update_xaxes(title_text="ρ = r/a", range=[0, 1])
    fig.update_yaxes(title_text="Valor normalizado", range=[0, None])
    fig.update_layout(height=420, margin=dict(t=40, b=20), legend=dict(
        orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5))
    st.plotly_chart(fig, use_container_width=True)

    # Perfiles D y v
    with st.expander("Ver perfiles D(ρ) y v(ρ)"):
        fig2 = make_subplots(rows=1, cols=2,
                             subplot_titles=["D(ρ) [m²/s]", "v(ρ) [m/s]"])
        fig2.add_trace(go.Scatter(x=RHO, y=D_prof, line=dict(color="#639922", width=2.5)),
                       row=1, col=1)
        fig2.add_trace(go.Scatter(x=RHO, y=v_prof, line=dict(color="#BA7517", width=2.5)),
                       row=1, col=2)
        fig2.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
        fig2.update_xaxes(title_text="ρ = r/a")
        fig2.update_layout(height=280, margin=dict(t=30, b=10), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

# ── Tab 2: Estados de carga ───────────────────────────────────────────────────
with tab2:
    if nz_mat is not None:
        st.caption("Distribución de densidad por estado de carga (Aurora)")
        # Mostrar solo los 10 estados con mayor integral
        totals = nz_mat.sum(axis=0)
        top_idx = np.argsort(totals)[::-1][:10]

        fig3 = go.Figure()
        palette = ["#185FA5","#0F6E56","#BA7517","#993C1D","#534AB7",
                   "#0B6E4F","#639922","#D4537E","#888780","#E24B4A"]
        for k, idx in enumerate(top_idx):
            if totals[idx] > 0:
                fig3.add_trace(go.Scatter(
                    x=RHO, y=nz_mat[:, idx] / max(nz_mat.max(), 1e-30),
                    name=cs_lbls[idx], line=dict(color=palette[k], width=2)
                ))
        fig3.update_xaxes(title_text="ρ = r/a", range=[0, 1])
        fig3.update_yaxes(title_text="n_z (norm.)")
        fig3.update_layout(height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("""
        **Activa Aurora** en el sidebar para ver la distribución de estados de carga.

        En modo analítico se usa un único fluido de impureza con PEC simplificado.
        Aurora calcula cada estado de carga por separado (W⁰ – W⁷⁴⁺) usando datos ADAS reales.
        """)
        # Mostrar balance de ionización coronal simplificado como proxy
        st.subheader("Balance coronal simplificado (aproximación)")
        Te_peak_pec = tr.SPECIES_PEC[species]["Te_peak"]
        sigma_pec   = tr.SPECIES_PEC[species]["sigma"]
        n_states = 8
        fig_cs = go.Figure()
        for z in range(n_states):
            Te_c = Te_peak_pec * (0.5 + z * 0.15)
            sig  = sigma_pec * 0.8
            frac = np.exp(-0.5 * ((Te_arr - Te_c) / sig)**2)
            frac /= (frac.max() + 1e-30)
            fig_cs.add_trace(go.Scatter(
                x=RHO, y=frac * nW_arr / nW_arr.max(),
                name=f"{species}{z+28}+",
                stackgroup="one", line=dict(width=0.5),
            ))
        fig_cs.update_xaxes(title_text="ρ = r/a", range=[0, 1])
        fig_cs.update_yaxes(title_text="Contribución relativa")
        fig_cs.update_layout(height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig_cs, use_container_width=True)
        st.caption("Vista esquemática: los estados de carga se distribuyen por temperatura. "
                   "Para valores exactos activa Aurora.")

# ── Tab 3: Mapa D–v ───────────────────────────────────────────────────────────
with tab3:
    st.subheader("Factor de pico ε(0)/ε(a) en el espacio D–v")
    st.caption("Calcula ~625 simulaciones — puede tardar ~5 segundos")

    if st.button("Calcular mapa", type="primary"):
        with st.spinner("Calculando mapa paramétrico…"):
            D_range = np.logspace(-2, 0.5, 25)
            v_range = np.linspace(-22, 22, 25)
            peak_map = tr.peaking_map(D_range, v_range, Te0=Te0, ne0=ne0_si, species=species)

        fig4 = go.Figure(go.Heatmap(
            x=np.log10(D_range),
            y=v_range,
            z=np.log10(np.clip(peak_map, 0.05, 50)),
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="log₁₀(η)", tickformat=".1f"),
            hovertemplate="D=10^%{x:.2f} m²/s<br>v=%{y:.1f} m/s<br>log₁₀(η)=%{z:.2f}<extra></extra>"
        ))

        # Marcar posición actual
        fig4.add_trace(go.Scatter(
            x=[np.log10(np.mean(D_prof))],
            y=[np.mean(v_prof)],
            mode="markers",
            marker=dict(symbol="star", size=18, color="yellow",
                        line=dict(color="black", width=1.5)),
            name="Configuración actual"
        ))

        fig4.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        fig4.update_xaxes(title_text="log₁₀(D)  [m²/s]",
                          tickvals=np.arange(-2, 1), ticktext=[f"10^{i}" for i in range(-2, 1)])
        fig4.update_yaxes(title_text="v [m/s]")
        fig4.update_layout(height=480, margin=dict(t=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.info("**Zona roja** (η > 1): perfil picado → W concentrado en el centro")
        with col_b:
            st.info("**Zona azul** (η < 1): perfil hueco → W principalmente en el borde")
    else:
        st.info("Presiona el botón para generar el mapa. "
                "La estrella marcará tu configuración actual del sidebar.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='font-size:12px;color:#888;text-align:center'>
Modelo analítico: ec. de transporte 1D estacionaria en geometría cilíndrica.
PEC simplificado (Gaussiana en T_e). Aurora: datos ADAS reales (aurora-fusion).
<br>Vicente Sepúlveda · PUC · Tesis EAST 2026
</div>
""", unsafe_allow_html=True)
