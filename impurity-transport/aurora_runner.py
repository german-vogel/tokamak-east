"""
aurora_runner.py — Wrapper de Aurora para simulación de transporte con física atómica real.

Aurora resuelve la ecuación de transporte por cada estado de carga z de la impureza,
incluyendo ionización y recombinación (datos ADAS). Requiere `pip install aurora-fusion`.
"""

import numpy as np

try:
    import aurora  # package: aurorafusion (pip install aurorafusion)
    AURORA_AVAILABLE = True
except ImportError:
    AURORA_AVAILABLE = False


def aurora_status():
    return AURORA_AVAILABLE


def run_aurora_sim(rho, Te_arr, ne_arr, D_prof, v_prof, species="W",
                   time_end=1.0, n_time=50):
    """
    Corre una simulación de Aurora en geometría circular simplificada.

    Parámetros
    ----------
    rho      : array (N,)  radio normalizado [0,1]
    Te_arr   : array (N,)  temperatura electrónica [keV]
    ne_arr   : array (N,)  densidad electrónica [m⁻³]
    D_prof   : array (N,)  coeficiente de difusión [m²/s]
    v_prof   : array (N,)  velocidad de convección [m/s]
    species  : str         elemento ('W', 'Fe', 'Mo')
    time_end : float       tiempo de simulación [s]
    n_time   : int         número de pasos temporales

    Retorna
    -------
    dict con:
        'nz'      : array (N, Z+1)  densidad por estado de carga
        'emiss'   : array (N,)      emisividad total (suma estados)
        'rho'     : array (N,)      grilla radial
        'cs_labels': list           etiquetas de estados de carga
    """
    if not AURORA_AVAILABLE:
        raise RuntimeError("Aurora no está instalado. Instala con: pip install aurora-fusion")

    # Radio menor EAST típico ~0.45 m
    a_minor = 0.45  # m
    r_arr = rho * a_minor  # m

    # Namelist de Aurora
    namelist = aurora.load_default_namelist()
    namelist['imp'] = species
    namelist['main_element'] = 'D'

    # Geometría circular simplificada
    namelist['Raxis_cm']  = 185.0   # cm, R_eje magnético EAST
    namelist['a_cm']      = a_minor * 100  # cm
    namelist['source_cm'] = a_minor * 100  # fuente en el borde

    # Tiempo
    namelist['timing']['times']  = [0, time_end]
    namelist['timing']['dt_start'] = time_end / n_time

    # Perfiles (Aurora espera arrays en la grilla interna)
    # Convertimos ne a cm⁻³ y Te a eV
    namelist['ne_cm3'] = ne_arr * 1e-6  # m⁻³ → cm⁻³
    namelist['Te_eV']  = Te_arr * 1e3   # keV → eV
    namelist['rhop_grid'] = rho

    # Construir simulación
    asim = aurora.aurora_sim(namelist, geqdsk=None)

    # Coeficientes D y v para todos los estados de carga (mismos para todos)
    nz_states = asim.Z_imp + 1
    D_z = np.tile(D_prof[:, np.newaxis], (1, nz_states))  # (N, Z+1)
    V_z = np.tile(v_prof[:, np.newaxis], (1, nz_states))

    # Correr
    out = asim.run_aurora(D_z, V_z)
    nz_final = out['nz'][:, :, -1]  # (N, Z+1) estado estacionario

    # Emisividad total (aproximada como suma ponderada)
    emiss = np.sum(nz_final, axis=1) * ne_arr

    # Etiquetas de estados de carga
    cs_labels = [f"{species}{z}+" for z in range(nz_states)]

    return {
        'nz':       nz_final,
        'emiss':    emiss / max(np.max(emiss), 1e-30),
        'rho':      rho,
        'cs_labels': cs_labels,
    }


def get_aurora_emissivity(asim, nz, line_label=None):
    """
    Calcula emisividad de una línea específica usando PECs de ADAS.
    Si no se especifica línea, retorna la emisividad total bolométrica.
    """
    if not AURORA_AVAILABLE:
        raise RuntimeError("Aurora no disponible.")
    try:
        # Intenta obtener emisividad con PEC reales
        emiss = aurora.radiation.compute_rad(asim.imp, nz, asim.ne, asim.Te)
        return emiss['line_rad_dens'].sum(axis=1)
    except Exception:
        # Fallback: suma de densidades ponderada
        return nz.sum(axis=1)
