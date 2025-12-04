import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI as CP

fluid = "Air"

# ================================
# FONCTIONS THERMO
# ================================
def get_v(T, P):
    """Volume spécifique du fluide"""
    return 1/CP("D","T",T,"P",P,fluid)

def get_h(T, P):
    """Enthalpie massique"""
    return CP("H","T",T,"P",P,fluid)

def get_s(T, P):
    """Entropie massique"""
    return CP("S","T",T,"P",P,fluid)

def isobar(P, T1, T2, n=200):
    """Transformation isobare entre deux températures"""
    T = np.linspace(T1,T2,n)
    V = [get_v(t,P) for t in T]
    s = [get_s(t,P) for t in T]
    return V, [P]*n, s, list(T)

def isentropic(S, P1, P2, n=200):
    """
    Transformation isentropique entre P1 et P2 pour entropie S
    Retourne V, P, s, T
    """
    P = np.linspace(P1,P2,n)
    T = [CP("T","P",p,"S",S,fluid) for p in P]
    V = [get_v(t,p) for t,p in zip(T,P)]
    s = [S]*n
    return V, P, s, T

# ================================
# FONCTION CYCLE BRAYTON
# ================================
def cycle_brayton(T1=300, P1=2e5, pressure_ratio=10, T3=1100, show_plot=True):
    """
    Calcule et trace le cycle de Brayton
    :param T1: Température entrée compresseur
    :param P1: Pression entrée compresseur
    :param pressure_ratio: rapport de pression compresseur
    :param T3: Température sortie chambre combustion
    :param show_plot: bool pour affichage
    :return: dictionnaire avec états, courbes et bilans énergétiques
    """

    # États clés
    P2 = P1 * pressure_ratio
    v1, h1, s1 = get_v(T1,P1), get_h(T1,P1), get_s(T1,P1)
    T2 = CP("T","P",P2,"S",s1,fluid)
    v2, h2, s2 = get_v(T2,P2), get_h(T2,P2), get_s(T2,P2)
    P3 = P2
    v3, h3, s3 = get_v(T3,P3), get_h(T3,P3), get_s(T3,P3)
    P4 = P1
    T4 = CP("T","P",P4,"S",s3,fluid)
    v4, h4, s4 = get_v(T4,P4), get_h(T4,P4), get_s(T4,P4)

    # Courbes
    V12, P12, s12, T12 = isentropic(s1,P1,P2)
    V23, P23, s23, T23 = isobar(P2,T2,T3)
    V34, P34, s34, T34 = isentropic(s3,P3,P4)
    V41, P41, s41, T41 = isobar(P1,T4,T1)

    # Bilan énergétique
    W_comp = h2 - h1
    W_turb = h3 - h4
    W_net = W_turb - W_comp
    Q_in = h3 - h2
    eta = W_net / Q_in

    results = {
        "V": [v1,v2,v3,v4],
        "P": [P1,P2,P3,P4],
        "h": [h1,h2,h3,h4],
        "s": [s1,s2,s3,s4],
        "curves": {
            "V12": V12, "P12": P12, "V23": V23, "P23": P23,
            "V34": V34, "P34": P34, "V41": V41, "P41": P41,
            "s12": s12, "s23": s23, "s34": s34, "s41": s41,
            "T12": T12, "T23": T23, "T34": T34, "T41": T41
        },
        "energetics": {"W_comp": W_comp, "W_turb": W_turb, "W_net": W_net, "Q_in": Q_in, "eta": eta}
    }

    if show_plot:
        plt.figure(figsize=(12,6))
        # P-V
        plt.subplot(1,2,1)
        plt.plot(V12,np.array(P12)/1e5,'r')
        plt.plot(V23,np.array(P23)/1e5,'b')
        plt.plot(V34,np.array(P34)/1e5,'g')
        plt.plot(V41,np.array(P41)/1e5,'m')
        plt.xlabel("Volume spécifique (m³/kg)")
        plt.ylabel("Pression (bar)")
        plt.title("Cycle de Brayton (P-V)")
        plt.grid()
        # T-S
        plt.subplot(1,2,2)
        plt.plot(s12,T12,'r')
        plt.plot(s23,T23,'b')
        plt.plot(s34,T34,'g')
        plt.plot(s41,T41,'m')
        plt.xlabel("Entropie massique (J/kg/K)")
        plt.ylabel("Température (K)")
        plt.title("Cycle de Brayton (T-S)")
        plt.grid()
        plt.tight_layout()
        plt.show()

        # Texte
        print("===== CYCLE DE BRAYTON =====")
        print(f"Travail net          : {W_net/1000:.2f} kJ/kg")
        print(f"Travail compresseur  : {W_comp/1000:.2f} kJ/kg")
        print(f"Travail turbine      : {W_turb/1000:.2f} kJ/kg")
        print(f"Chaleur reçue        : {Q_in/1000:.2f} kJ/kg")
        print(f"Rendement théorique  : {eta*100:.2f} %")

    return results

if __name__=="__main__":
    cycle_brayton()
