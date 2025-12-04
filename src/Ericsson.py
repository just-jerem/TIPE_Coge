import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI as CP
import warnings

fluid = "Air"  # fluide de travail

# ================================
# FONCTIONS THERMO
# ================================
def get_v(T, P):
    rho = CP("D", "T", T, "P", P, fluid)
    return 1 / rho

def get_h(T, P):
    return CP("H", "T", T, "P", P, fluid)

def get_s(T, P):
    return CP("S", "T", T, "P", P, fluid)

def isotherm(T, P1, P2, n=200):
    P = np.linspace(P1, P2, n)
    V = [get_v(T, p) for p in P]
    s = [get_s(T, p) for p in P]
    T_list = [T]*n
    return V, P, s, T_list

def isobar(P, T1, T2, n=200):
    T = np.linspace(T1, T2, n)
    V = [get_v(t, P) for t in T]
    s = [get_s(t, P) for t in T]
    return V, [P]*n, s, list(T)

def _integrate_PdV(V_list, P_list):
    """
    Intégration numérique ∫ P dV (J/kg).
    S'assure que V est croissant pour trapz (sinon inverse les tableaux).
    """
    V = np.array(V_list)
    P = np.array(P_list)
    if V[0] > V[-1]:
        V = V[::-1]
        P = P[::-1]
    return np.trapz(P, V)

# ================================
# FONCTION CYCLE ERICSSON (CORRIGÉE)
# ================================
def cycle_ericsson(T_min=300, T_max=1100, P_min=2e5, P_max=20e5, show_plot=True):
    """
    Calcule et trace le cycle d'Ericsson (version robuste)
    :param T_min: Température basse (K)
    :param T_max: Température haute (K)
    :param P_min: Pression basse (Pa)
    :param P_max: Pression haute (Pa)
    :param show_plot: bool, True pour afficher graphiques et impressions
    :return: dictionnaire avec états, courbes et bilans énergétiques
    """

    # -----------------
    # États clés du cycle
    # -----------------
    V1 = get_v(T_min, P_min)
    V2 = get_v(T_min, P_max)
    V3 = get_v(T_max, P_max)
    V4 = get_v(T_max, P_min)

    s1 = get_s(T_min, P_min)
    s2 = get_s(T_min, P_max)
    s3 = get_s(T_max, P_max)
    s4 = get_s(T_max, P_min)

    h1 = get_h(T_min, P_min)
    h2 = get_h(T_min, P_max)
    h3 = get_h(T_max, P_max)
    h4 = get_h(T_max, P_min)

    # -----------------
    # Courbes du cycle
    # -----------------
    V12, P12, s12, T12 = isotherm(T_min, P_min, P_max)   # isotherme froide (compression 1->2)
    V23, P23, s23, T23 = isobar(P_max, T_min, T_max)     # isobare chauffage/regénération 2->3
    V34, P34, s34, T34 = isotherm(T_max, P_max, P_min)   # isotherme chaude (expansion 3->4)
    V41, P41, s41, T41 = isobar(P_min, T_max, T_min)     # isobare refroidissement/regénération 4->1

    # -----------------
    # Bilan énergétique (CORRECT)
    # -----------------
    # Chaleur reçue durant l'expansion isotherme 3->4 (doit être positive)
    ds_hot = s4 - s3
    Q_in = T_max * ds_hot    # J/kg

    # Chaleur rejetée durant la compression isotherme 1->2
    # ATTENTION : s2 - s1 < 0 typiquement (augmentation de pression, entropie diminue).
    # La chaleur rejetée (valeur positive) = T_min * (s1 - s2)
    ds_cold = s2 - s1
    Q_out = T_min * (s1 - s2)   # = -T_min*(s2 - s1)

    # Travail net et rendement
    W_cycle = Q_in - Q_out
    eps = 1e-12
    if abs(Q_in) < eps:
        warnings.warn("Q_in très proche de 0 : vérifier T_max et états 3->4")
    eta = W_cycle / Q_in if abs(Q_in) > eps else np.nan

    # -----------------
    # Vérifications numériques (intégration P dV sur isothermes)
    # -----------------
    W34 = _integrate_PdV(V34, P34)   # travail isotherme d'expansion (devrait ~ Q_in)
    W12 = _integrate_PdV(V12, P12)   # travail isotherme de compression (devrait ~ Q_out)

    # W12 calculé par ∫P dV donne la valeur algébrique du travail (compression -> positif si orientation V inc),
    # mais on veut la valeur de chaleur rejetée positive; on compare en valeur absolue.
    rel_diff_in = abs(W34 - Q_in) / (abs(Q_in) + 1e-12)
    rel_diff_out = abs(W12 - Q_out) / (abs(Q_out) + 1e-12)

    if rel_diff_in > 1e-2:
        warnings.warn(f"Écart relatif Q_in vs ∫P dV (3->4) = {rel_diff_in:.3e}")
    if rel_diff_out > 1e-2:
        warnings.warn(f"Écart relatif Q_out vs ∫P dV (1->2) = {rel_diff_out:.3e}")

    # Ajustements numériquement prudents si les deux méthodes sont cohérentes
    if rel_diff_in < 1e-2 and rel_diff_out < 1e-2:
        Q_in = 0.5*(Q_in + W34)
        Q_out = 0.5*(Q_out + W12)
        W_cycle = Q_in - Q_out
        eta = W_cycle / Q_in if abs(Q_in) > eps else np.nan
    else:
        # si grosse divergence, on émet un avertissement mais on continue avec TΔs
        warnings.warn("Divergence notable entre méthodes TΔs et ∫P dV ; conserver TΔs pour Q.")

    # Clamp numérique : corrige petits dépassements dus au bruit
    if eta > 1 and eta < 1.001:
        warnings.warn(f"eta légèrement > 1 ({eta}); clamp à 1.0")
        eta = 1.0

    if eta < -0.01:
        # rendement négatif important : alerte
        warnings.warn(f"Rendement négatif détecté (eta={eta}). Vérifier paramètres.")

    # Si rendement manifestement >1, on lève une erreur pour forcer débogage
    if eta > 1.0 + 1e-9:
        raise ValueError(f"Rendement > 1 détecté ({eta}). Vérifier définitions d'états et signes.")

    # -----------------
    # Organisation des résultats
    # -----------------
    results = {
        "V": [V1, V2, V3, V4],
        "P": [P_min, P_max, P_max, P_min],
        "h": [h1, h2, h3, h4],
        "s": [s1, s2, s3, s4],
        "curves": {
            "V12": V12, "P12": P12, "V23": V23, "P23": P23,
            "V34": V34, "P34": P34, "V41": V41, "P41": P41,
            "s12": s12, "s23": s23, "s34": s34, "s41": s41,
            "T12": T12, "T23": T23, "T34": T34, "T41": T41
        },
        "energetics": {"Q_in": Q_in, "Q_out": Q_out, "W_cycle": W_cycle, "eta": eta},
        "debug": {
            "ds_hot": ds_hot, "ds_cold": ds_cold,
            "W34": W34, "W12": W12,
            "rel_diff_in": rel_diff_in, "rel_diff_out": rel_diff_out
        }
    }

    # -----------------
    # Affichage graphique et texte si demandé
    # -----------------
    if show_plot:
        plt.figure(figsize=(12,6))

        # Diagramme P-V
        plt.subplot(1,2,1)
        plt.plot(V12, np.array(P12)/1e5,'r', label="1→2 Isotherme")
        plt.plot(V23, np.array(P23)/1e5,'b', label="2→3 Isobare")
        plt.plot(V34, np.array(P34)/1e5,'g', label="3→4 Isotherme")
        plt.plot(V41, np.array(P41)/1e5,'m', label="4→1 Isobare")
        plt.scatter([V1,V2,V3,V4],[P_min/1e5,P_max/1e5,P_max/1e5,P_min/1e5], color='k')
        plt.xlabel("Volume spécifique (m³/kg)")
        plt.ylabel("Pression (bar)")
        plt.title("Cycle d'Ericsson (P-V)")
        plt.grid()
        plt.legend()

        # Diagramme T-S
        plt.subplot(1,2,2)
        plt.plot(s12, T12,'r', label="1→2 Isotherme")
        plt.plot(s23, T23,'b', label="2→3 Isobare")
        plt.plot(s34, T34,'g', label="3→4 Isotherme")
        plt.plot(s41, T41,'m', label="4→1 Isobare")
        plt.scatter([s1,s2,s3,s4],[T_min,T_min,T_max,T_max], color='k')
        plt.xlabel("Entropie massique (J/kg/K)")
        plt.ylabel("Température (K)")
        plt.title("Cycle d'Ericsson (T-S)")
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()

        # affichage texte
        print("===== CYCLE D'ERICSSON =====")
        print(f"Δs chaude (s4-s3) : {ds_hot:.6e} J/kg/K")
        print(f"Δs froide (s2-s1) : {ds_cold:.6e} J/kg/K")
        print(f"Q_in (J/kg)       : {Q_in:.6e}")
        print(f"Q_out (J/kg)      : {Q_out:.6e}")
        print(f"W_cycle (J/kg)    : {W_cycle:.6e}")
        print(f"eta               : {eta:.6f}")

    return results

# ================================
# Test direct
# ================================
if __name__ == "__main__":
    cycle_ericsson()
