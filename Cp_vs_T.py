"""
Tracé de Cp(T) pour l'air — CoolProp vs. gaz parfait
Sauvegarde : fig/Cp_vs_T.pdf et fig/Cp_vs_T.png
"""

import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

# ── Paramètres ────────────────────────────────────────────────
T_min, T_max = 300, 1300          # plage de températures (K)
N = 300                            # points de discrétisation
pressures_bar = [1, 5, 10, 20]    # pressions (bar)
Cp_gp = 1004.5                     # gaz parfait référence ISO (J/kg/K)

# ── Calcul CoolProp ───────────────────────────────────────────
T = np.linspace(T_min, T_max, N)

fig, ax = plt.subplots(figsize=(10, 5.5))
colors = plt.cm.Blues(np.linspace(0.4, 1.0, len(pressures_bar)))

for P_bar, color in zip(pressures_bar, colors):
    P_Pa = P_bar * 1e5
    Cp = np.array([PropsSI("Cpmass", "T", Ti, "P", P_Pa, "Air") for Ti in T])
    ax.plot(T, Cp, color=color, lw=2.0, label=f"CoolProp — {P_bar} bar")

# Référence gaz parfait
ax.axhline(Cp_gp, color="crimson", lw=1.8, ls="--",
           label=f"Gaz parfait — $C_p = {Cp_gp:.0f}$ J/(kg·K)")

# ── Annotations températures d'intérêt ───────────────────────
for T_mark, label in [(700, "700 K"), (1100, "1100 K")]:
    ax.axvline(T_mark, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.text(T_mark + 10, ax.get_ylim()[0] + 5 if ax.get_ylim()[0] > 0 else 1010,
            label, fontsize=8, color="gray", va="bottom")

# ── Mise en forme ─────────────────────────────────────────────
ax.set_xlabel("Température $T$ (K)", fontsize=12)
ax.set_ylabel(r"$C_p$ (J·kg$^{-1}$·K$^{-1}$)", fontsize=12)
ax.set_title(r"Capacité thermique massique $C_p(T,P)$ de l'air", fontsize=13)
ax.set_xlim(T_min, T_max)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, alpha=0.3)

# Écart relatif à 1100 K pour annotation
P1_Pa = 1e5
Cp_1100_cool = PropsSI("Cpmass", "T", 1100, "P", P1_Pa, "Air")
ecart = (Cp_1100_cool - Cp_gp) / Cp_gp * 100
ax.annotate(f"+{ecart:.1f} % à 1100 K\n(1 bar vs. GP)",
            xy=(1100, Cp_1100_cool), xytext=(950, Cp_1100_cool + 20),
            fontsize=8, color=colors[-1],
            arrowprops=dict(arrowstyle="->", color=colors[-1], lw=1.0))

plt.tight_layout()

import os
os.makedirs("fig", exist_ok=True)
plt.savefig("fig/Cp_vs_T.pdf", dpi=150)
plt.savefig("fig/Cp_vs_T.png", dpi=150)
print(f"Figures sauvegardées : fig/Cp_vs_T.pdf / .png")
print(f"Écart CoolProp vs. GP à 1100 K, 1 bar : +{ecart:.1f} %")
plt.show()
