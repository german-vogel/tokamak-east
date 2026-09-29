"""
transport.py — Modelo analítico de transporte de impurezas 1D

Solución exacta de la ecuación estacionaria en geometría cilíndrica:
    (1/r) d/dr [ r (D dn/dr - v·n) ] = 0

con condición de regularidad en r=0 → D dn/dr = v·n → n(ρ) = n_edge·exp(∫_ρ^1 v/D dρ')
"""

import numpy as np

# Compatibilidad NumPy 1.x (trapz) y 2.x (trapezoid)
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz
from scipy.interpolate import interp1d

RHO = np.linspace(0.0, 1.0, 300)


# ── Perfiles de plasma ────────────────────────────────────────────────────────

def plasma_profiles(rho, Te0=3.0, Te_edge=0.1, ne0=3e19, ne_edge=5e18,
                    alpha_Te=1.5, alpha_ne=1.0):
    Te = Te0 * (1 - rho**2)**alpha_Te + Te_edge
    ne = ne0 * (1 - rho**2)**alpha_ne + ne_edge
    return Te, ne


# ── PEC simplificado (Gaussiana en Te) ───────────────────────────────────────

SPECIES_PEC = {
    "W":  {"Te_peak": 1.5, "sigma": 0.7, "label": "W (EUV ~1.5 keV)"},
    "Fe": {"Te_peak": 0.8, "sigma": 0.4, "label": "Fe (EUV ~0.8 keV)"},
    "Mo": {"Te_peak": 1.2, "sigma": 0.5, "label": "Mo (EUV ~1.2 keV)"},
}

def pec_approx(Te_arr, Te_peak=1.5, sigma=0.7):
    return np.exp(-0.5 * ((Te_arr - Te_peak) / sigma)**2)


# ── Transporte analítico ──────────────────────────────────────────────────────

def solve_transport(rho, D_prof, v_prof, n_edge=1.0):
    """
    Perfil de densidad de impureza (normalizado a n_edge en ρ=1).
    Integra v/D desde cada ρ hasta el borde.
    """
    vD = v_prof / np.maximum(D_prof, 1e-6)
    n = np.zeros_like(rho)
    for i in range(len(rho)):
        integral = _trapz(vD[i:], rho[i:])
        n[i] = n_edge * np.exp(np.clip(-integral, -50, 50))  # clip evita overflow numérico
    return n


def emissivity_profile(rho, n_imp, ne_arr, Te_arr, species="W"):
    """Emisividad ε ∝ n_imp · ne · PEC(Te), normalizada al máximo."""
    p = SPECIES_PEC[species]
    pec = pec_approx(Te_arr, p["Te_peak"], p["sigma"])
    eps = n_imp * ne_arr * pec
    mx = np.max(eps)
    return eps / mx if mx > 0 else eps


def peaking_factor(eps):
    """ε(0) / ε(a) — factor de concentración central."""
    return eps[0] / max(eps[-1], 1e-12)


# ── Editor de perfiles D(ρ) / v(ρ) ──────────────────────────────────────────

CTRL_POINTS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

def build_profile(rho, ctrl_values, kind="cubic"):
    """Interpola valores en puntos de control a la grilla completa."""
    f = interp1d(CTRL_POINTS, ctrl_values, kind=kind, fill_value="extrapolate")
    return np.clip(f(rho), 1e-6, None)


# ── Mapa de pico 2D ──────────────────────────────────────────────────────────

def peaking_map(D_range, v_range, Te0=3.0, ne0=3e19, species="W"):
    """Calcula factor de pico en grilla D×v."""
    Te, ne = plasma_profiles(RHO, Te0=Te0, ne0=ne0)
    peak = np.zeros((len(v_range), len(D_range)))
    for i, v in enumerate(v_range):
        for j, D in enumerate(D_range):
            D_p = np.full_like(RHO, D)
            v_p = np.full_like(RHO, v)
            nW = solve_transport(RHO, D_p, v_p)
            eps = emissivity_profile(RHO, nW, ne, Te, species)
            peak[i, j] = peaking_factor(eps)
    return peak
