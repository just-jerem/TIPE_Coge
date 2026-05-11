"""
Graphe comparatif : Ericsson gaz parfait vs CoolProp
Deux sous-graphes côte à côte :
  - Gauche : η en fonction de Tmax (rp et Tmin fixés)
  - Droite  : η en fonction de rp  (Tmax et Tmin fixés)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from CoolProp.CoolProp import PropsSI as CP
from functools import lru_cache

# ── Paramètres ────────────────────────────────────────────────
T_MIN   = 300       # K  (fixe)
T_MAX   = 1100      # K  (fixe pour le graphe de droite)
P_MIN   = 1e5       # Pa = 1 bar (fixe)
RP_REF  = 10        # rapport de pression de référence (graphe gauche)

T_MAX_RANGE = np.linspace(600, 1300, 120)   # abscisse gauche
RP_RANGE    = np.linspace(1.5, 25,   120)   # abscisse droite

FLUID = "Air"

# ── CoolProp avec cache ───────────────────────────────────────
@lru_cache(maxsize=50000)
def get_s(T, P):
    return CP("S", "T", float(T), "P", float(P), FLUID)

# ── Ericsson CoolProp ─────────────────────────────────────────
def eta_ericsson_coolprop(T_min, T_max, P_min, rp):
    P_max = P_min * rp
    s1 = get_s(T_min, P_min)
    s2 = get_s(T_min, P_max)
    s3 = get_s(T_max, P_max)
    s4 = get_s(T_max, P_min)
    Q_in  = T_max * (s4 - s3)
    Q_out = T_min * (s1 - s2)
    W_net = Q_in - Q_out
    return W_net / Q_in

# ── Ericsson gaz parfait ──────────────────────────────────────
# η_GP = 1 - Tmin/Tmax  (indépendant de rp avec régénération)
def eta_ericsson_gp(T_min, T_max):
    return 1.0 - T_min / T_max

# ── Calcul des courbes ────────────────────────────────────────
# Graphe gauche : f(Tmax), rp fixé
eta_cp_tmax  = [eta_ericsson_coolprop(T_MIN, T, P_MIN, RP_REF) * 100
                for T in T_MAX_RANGE]
eta_gp_tmax  = [eta_ericsson_gp(T_MIN, T) * 100
                for T in T_MAX_RANGE]
eta_carnot_tmax = [eta_ericsson_gp(T_MIN, T) * 100   # = Carnot
                   for T in T_MAX_RANGE]

# Graphe droit : f(rp), Tmax fixé
# Pour gaz parfait : η indépendant de rp → ligne horizontale
eta_cp_rp   = [eta_ericsson_coolprop(T_MIN, T_MAX, P_MIN, rp) * 100
               for rp in RP_RANGE]
eta_gp_rp   = [eta_ericsson_gp(T_MIN, T_MAX) * 100] * len(RP_RANGE)

# ── Mise en page ──────────────────────────────────────────────
BLEU    = "#1A3A6B"   # bleuMarine
ROUGE   = "#C0392B"   # rougeCycle
ORANGE  = "#D36900"   # orangeAcc
GRIS    = "#888888"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.subplots_adjust(wspace=0.38, left=0.09, right=0.97,
                    top=0.88, bottom=0.13)

# ── Sous-graphe gauche ────────────────────────────────────────
ax = axes[0]
ax.plot(T_MAX_RANGE, eta_cp_tmax,
        color=BLEU,  lw=2.2, label="Ericsson — CoolProp (gaz réel)")
ax.plot(T_MAX_RANGE, eta_gp_tmax,
        color=ROUGE, lw=2.2, linestyle="--",
        label="Ericsson — Gaz parfait ($\\eta = 1 - T_{\\min}/T_{\\max}$)")

# Annotation de l'écart à Tmax = 1100 K
T_ref = 1100
eta_cp_ref  = eta_ericsson_coolprop(T_MIN, T_ref, P_MIN, RP_REF) * 100
eta_gp_ref  = eta_ericsson_gp(T_MIN, T_ref) * 100
ecart = eta_gp_ref - eta_cp_ref
ax.annotate("",
    xy=(T_ref, eta_cp_ref), xytext=(T_ref, eta_gp_ref),
    arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.8))
ax.text(T_ref + 18, (eta_cp_ref + eta_gp_ref) / 2,
        f"$\\Delta\\eta = {ecart:.1f}$ pts",
        color=ORANGE, fontsize=9, va="center")

ax.set_xlabel("$T_{\\max}$ (K)", fontsize=11)
ax.set_ylabel("$\\eta$ (%)", fontsize=11)
ax.set_title(f"Influence de $T_{{\\max}}$\n"
             f"$T_{{\\min}}={T_MIN}\\,\\mathrm{{K}}$, "
             f"$r_p={RP_REF}$",
             fontsize=10)
ax.set_xlim(T_MAX_RANGE[0], T_MAX_RANGE[-1])
ax.set_ylim(45, 82)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(True, which="major", linestyle="--", alpha=0.4)
ax.grid(True, which="minor", linestyle=":",  alpha=0.2)
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)

# ── Sous-graphe droit ─────────────────────────────────────────
ax = axes[1]
ax.plot(RP_RANGE, eta_cp_rp,
        color=BLEU,  lw=2.2, label="Ericsson — CoolProp (gaz réel)")
ax.plot(RP_RANGE, eta_gp_rp,
        color=ROUGE, lw=2.2, linestyle="--",
        label="Ericsson — Gaz parfait (constant)")

# Annotation : η GP est constant car indépendant de rp
ax.text(RP_RANGE[-1] - 1.5, eta_gp_rp[0] + 0.5,
        "GP : $\\eta$ indép. de $r_p$",
        color=ROUGE, fontsize=8.5, ha="right")

# Annotation écart à rp = 10
rp_ref_idx = np.argmin(np.abs(RP_RANGE - RP_REF))
eta_cp_rp10  = eta_cp_rp[rp_ref_idx]
eta_gp_rp10  = eta_gp_rp[rp_ref_idx]
ecart_rp = eta_gp_rp10 - eta_cp_rp10
ax.annotate("",
    xy=(RP_REF, eta_cp_rp10), xytext=(RP_REF, eta_gp_rp10),
    arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.8))
ax.text(RP_REF + 0.6, (eta_cp_rp10 + eta_gp_rp10) / 2,
        f"$\\Delta\\eta = {ecart_rp:.1f}$ pts\n($r_p={RP_REF}$)",
        color=ORANGE, fontsize=9, va="center")

ax.set_xlabel("$r_p = P_{\\max}/P_{\\min}$", fontsize=11)
ax.set_ylabel("$\\eta$ (%)", fontsize=11)
ax.set_title(f"Influence de $r_p$\n"
             f"$T_{{\\min}}={T_MIN}\\,\\mathrm{{K}}$, "
             f"$T_{{\\max}}={T_MAX}\\,\\mathrm{{K}}$",
             fontsize=10)
ax.set_xlim(RP_RANGE[0], RP_RANGE[-1])
ax.set_ylim(60, 76)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(True, which="major", linestyle="--", alpha=0.4)
ax.grid(True, which="minor", linestyle=":",  alpha=0.2)
ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)

# ── Titre global ──────────────────────────────────────────────
fig.suptitle(
    "Ericsson : comparaison gaz parfait vs gaz réel (CoolProp)\n"
    r"$P_{\min} = 1\,\mathrm{bar}$, fluide : Air",
    fontsize=11, y=0.98
)

plt.savefig("gp_vs_coolprop.pdf", dpi=300, bbox_inches="tight")
plt.savefig("gp_vs_coolprop.png", dpi=200, bbox_inches="tight")
print("Figures sauvegardées : gp_vs_coolprop.pdf / .png")

# ── Affichage des valeurs clés ────────────────────────────────
print(f"\nValeurs de référence (Tmax={T_ref} K, rp={RP_REF}) :")
print(f"  η gaz parfait = {eta_gp_ref:.2f} %")
print(f"  η CoolProp    = {eta_cp_ref:.2f} %")
print(f"  Écart         = {ecart:.2f} points")