"""
Cycle de Stirling - Simulation thermodynamique
===============================================
Le cycle de Stirling est composé de :
- 2 transformations isothermes (compression à T_min, détente à T_max)
- 2 transformations isochores (chauffage et refroidissement à volume constant)

Points du cycle :
1 → 2 : Compression isotherme à T_min (V1 → V2, chaleur rejetée)
2 → 3 : Chauffage isochore à V2 (T_min → T_max)
3 → 4 : Détente isotherme à T_max (V2 → V1, chaleur absorbée)
4 → 1 : Refroidissement isochore à V1 (T_max → T_min)
"""

import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI as CP
import warnings
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import os

fluid = "Air"  # fluide de travail
NUM_WORKERS = min(4, os.cpu_count() or 1)

# ================================
# CACHE POUR COOLPROP
# ================================
@lru_cache(maxsize=50000)
def CP_cached(output, input1, val1, input2, val2, fluid_name):
    """Cache les appels CoolProp pour améliorer les performances"""
    return CP(output, input1, val1, input2, val2, fluid_name)

# ================================
# FONCTIONS THERMODYNAMIQUES
# ================================
def get_v(T, P):
    """Volume spécifique (m³/kg) à partir de T et P"""
    rho = CP_cached("D", "T", T, "P", P, fluid)
    return 1 / rho

def get_h(T, P):
    """Enthalpie massique (J/kg) à partir de T et P"""
    return CP_cached("H", "T", T, "P", P, fluid)

def get_s(T, P):
    """Entropie massique (J/kg/K) à partir de T et P"""
    return CP_cached("S", "T", T, "P", P, fluid)

def get_P_from_TV(T, V):
    """Pression (Pa) à partir de T et V (volume spécifique)"""
    rho = 1 / V
    return CP_cached("P", "T", T, "D", rho, fluid)

# ================================
# TRANSFORMATIONS DU CYCLE
# ================================
def isotherm(T, V1, V2, n=200):
    """
    Transformation isotherme à température T constante
    Retourne V, P, s, T pour la courbe
    """
    V = np.linspace(V1, V2, n)
    
    def get_props(v):
        P = get_P_from_TV(T, v)
        s = get_s(T, P)
        return P, s
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(get_props, V))
    
    P_list = [r[0] for r in results]
    s_list = [r[1] for r in results]
    T_list = [T] * n
    
    return list(V), P_list, s_list, T_list

def isochore(V, T1, T2, n=200):
    """
    Transformation isochore à volume V constant
    Retourne V, P, s, T pour la courbe
    """
    T = np.linspace(T1, T2, n)
    
    def get_props(t):
        P = get_P_from_TV(t, V)
        s = get_s(t, P)
        return P, s
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(get_props, T))
    
    P_list = [r[0] for r in results]
    s_list = [r[1] for r in results]
    V_list = [V] * n
    
    return V_list, P_list, s_list, list(T)

def _integrate_PdV(V_list, P_list):
    """
    Intégration numérique ∫ P dV (J/kg).
    S'assure que V est croissant pour trapezoid (sinon inverse les tableaux).
    """
    V = np.array(V_list)
    P = np.array(P_list)
    if V[0] > V[-1]:
        V = V[::-1]
        P = P[::-1]
    return np.trapezoid(P, V)

# ================================
# FONCTION CYCLE STIRLING
# ================================
def cycle_stirling(T_min=300, T_max=1100, V_min=0.3, V_max=1.0, show_plot=True):
    """
    Calcule et trace le cycle de Stirling
    
    :param T_min: Température basse (K) - compression isotherme
    :param T_max: Température haute (K) - détente isotherme
    :param V_min: Volume spécifique minimal (m³/kg) - fin de compression
    :param V_max: Volume spécifique maximal (m³/kg) - fin de détente
    :param show_plot: bool, True pour afficher graphiques et impressions
    :return: dictionnaire avec états, courbes et bilans énergétiques
    """
    
    # -----------------
    # États clés du cycle
    # Point 1 : V_max, T_min (début compression isotherme)
    # Point 2 : V_min, T_min (fin compression, début chauffage)
    # Point 3 : V_min, T_max (fin chauffage, début détente)
    # Point 4 : V_max, T_max (fin détente, début refroidissement)
    # -----------------
    
    V1 = V_max
    V2 = V_min
    V3 = V_min
    V4 = V_max
    
    # Calcul des pressions aux 4 points
    P1 = get_P_from_TV(T_min, V1)
    P2 = get_P_from_TV(T_min, V2)
    P3 = get_P_from_TV(T_max, V3)
    P4 = get_P_from_TV(T_max, V4)
    
    # Enthalpies et entropies
    h1 = get_h(T_min, P1)
    h2 = get_h(T_min, P2)
    h3 = get_h(T_max, P3)
    h4 = get_h(T_max, P4)
    
    s1 = get_s(T_min, P1)
    s2 = get_s(T_min, P2)
    s3 = get_s(T_max, P3)
    s4 = get_s(T_max, P4)
    
    # -----------------
    # Courbes du cycle
    # -----------------
    V12, P12, s12, T12 = isotherm(T_min, V1, V2)    # isotherme froide (compression 1→2)
    V23, P23, s23, T23 = isochore(V2, T_min, T_max)  # isochore chauffage 2→3
    V34, P34, s34, T34 = isotherm(T_max, V3, V4)    # isotherme chaude (détente 3→4)
    V41, P41, s41, T41 = isochore(V1, T_max, T_min)  # isochore refroidissement 4→1
    
    # -----------------
    # Bilan énergétique
    # -----------------
    
    # Travail isotherme compression 1→2 (négatif car compression)
    W_12 = _integrate_PdV(V12, P12)
    
    # Travail isotherme détente 3→4 (positif car détente)
    W_34 = _integrate_PdV(V34, P34)
    
    # Pas de travail sur les isochores (dV = 0)
    W_23 = 0
    W_41 = 0
    
    # Chaleur échangée sur isothermes (Q = W pour isotherme gaz parfait)
    # Compression isotherme : chaleur rejetée Q_out = -W_12 (W_12 < 0, donc Q_out > 0)
    # Détente isotherme : chaleur absorbée Q_in = W_34 (W_34 > 0)
    
    # Pour un gaz réel, on utilise Q = T * ΔS
    Q_in = T_max * (s4 - s3)    # Chaleur absorbée pendant détente isotherme
    Q_out = T_min * (s1 - s2)   # Chaleur rejetée pendant compression isotherme
    
    # Chaleur sur isochores (récupérée par régénérateur idéal)
    Q_23 = h3 - h2  # Chaleur de chauffage (isochore)
    Q_41 = h1 - h4  # Chaleur de refroidissement (isochore)
    
    # Travail net du cycle
    W_cycle = W_34 + W_12  # W_12 est négatif
    
    # Vérification avec premier principe
    W_cycle_check = Q_in - Q_out
    
    # Rendement
    eps = 1e-12
    if abs(Q_in) < eps:
        warnings.warn("Q_in très proche de 0 : vérifier paramètres")
    eta = W_cycle_check / Q_in if abs(Q_in) > eps else np.nan
    
    # Rendement de Carnot théorique
    eta_carnot = 1 - T_min / T_max
    
    # Clamp numérique
    if eta > 1 and eta < 1.001:
        warnings.warn(f"eta légèrement > 1 ({eta}); clamp à 1.0")
        eta = 1.0
    
    if eta < -0.01:
        warnings.warn(f"Rendement négatif détecté (eta={eta}). Vérifier paramètres.")
    
    # -----------------
    # Organisation des résultats
    # -----------------
    results = {
        "V": [V1, V2, V3, V4],
        "P": [P1, P2, P3, P4],
        "h": [h1, h2, h3, h4],
        "s": [s1, s2, s3, s4],
        "curves": {
            "V12": V12, "P12": P12, "V23": V23, "P23": P23,
            "V34": V34, "P34": P34, "V41": V41, "P41": P41,
            "s12": s12, "s23": s23, "s34": s34, "s41": s41,
            "T12": T12, "T23": T23, "T34": T34, "T41": T41
        },
        "energetics": {
            "Q_in": Q_in, 
            "Q_out": Q_out, 
            "W_cycle": W_cycle_check,
            "Q_regen": Q_23,  # Chaleur régénérateur
            "eta": eta,
            "eta_carnot": eta_carnot
        },
        "debug": {
            "W_12": W_12, "W_34": W_34,
            "Q_23": Q_23, "Q_41": Q_41,
            "W_cycle_PdV": W_cycle
        }
    }
    
    # -----------------
    # Affichage graphique et texte si demandé
    # -----------------
    if show_plot:
        plt.figure(figsize=(12, 6))
        
        # Diagramme P-V
        plt.subplot(1, 2, 1)
        plt.plot(V12, np.array(P12)/1e5, 'r', label="1→2 Isotherme (compression)")
        plt.plot(V23, np.array(P23)/1e5, 'b', label="2→3 Isochore (chauffage)")
        plt.plot(V34, np.array(P34)/1e5, 'g', label="3→4 Isotherme (détente)")
        plt.plot(V41, np.array(P41)/1e5, 'm', label="4→1 Isochore (refroidissement)")
        plt.scatter([V1, V2, V3, V4], [P1/1e5, P2/1e5, P3/1e5, P4/1e5], color='k', zorder=5)
        plt.xlabel("Volume spécifique (m³/kg)")
        plt.ylabel("Pression (bar)")
        plt.title("Cycle de Stirling (P-V)")
        plt.grid()
        plt.legend()
        
        # Diagramme T-S
        plt.subplot(1, 2, 2)
        plt.plot(s12, T12, 'r', label="1→2 Isotherme")
        plt.plot(s23, T23, 'b', label="2→3 Isochore")
        plt.plot(s34, T34, 'g', label="3→4 Isotherme")
        plt.plot(s41, T41, 'm', label="4→1 Isochore")
        plt.scatter([s1, s2, s3, s4], [T_min, T_min, T_max, T_max], color='k', zorder=5)
        plt.xlabel("Entropie massique (J/kg/K)")
        plt.ylabel("Température (K)")
        plt.title("Cycle de Stirling (T-S)")
        plt.grid()
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Affichage texte
        print("===== CYCLE DE STIRLING =====")
        print(f"Q_in (J/kg)       : {Q_in:.6e}")
        print(f"Q_out (J/kg)      : {Q_out:.6e}")
        print(f"W_cycle (J/kg)    : {W_cycle_check:.6e}")
        print(f"Q_regen (J/kg)    : {Q_23:.6e}")
        print(f"eta               : {eta:.6f}")
        print(f"eta_Carnot        : {eta_carnot:.6f}")
    
    return results

# ================================
# Test direct
# ================================
if __name__ == "__main__":
    cycle_stirling()
