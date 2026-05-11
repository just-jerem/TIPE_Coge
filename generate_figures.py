"""
Génération des figures pour la présentation TIPE.
Utilise les modules existants src/Brayton.py, src/Ericsson.py, src/Stirling.py.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("fig", exist_ok=True)

# ── Palette identique à la présentation beamer ──────────────
C_B = (192/255,  57/255,  43/255)   # rougeCycle
C_E = ( 26/255,  58/255, 107/255)   # bleuMarine
C_S = ( 30/255, 132/255,  73/255)   # vertCycle
C_A = (211/255, 105/255,   0/255)   # orangeAcc
C_D = ( 50/255,  50/255,  60/255)   # grisTexte

plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "lines.linewidth": 1.6,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
})
SEG_C = [C_B, C_E, C_S, C_A]

def save(name):
    plt.tight_layout()
    plt.savefig(f"fig/{name}", bbox_inches="tight")
    plt.close()
    print(f"  ✓ fig/{name}")

# ============================================================
# 1. Importer et calculer les 3 cycles
# ============================================================
from src import Brayton, Ericsson, Stirling

TMIN, TMAX = 300.0, 1100.0
PMIN, PMAX = 2e5, 20e5
VMIN, VMAX = 0.3, 1.0
RP = 10

print("Calcul des cycles (peut prendre ~30 s)...")
rb = Brayton.cycle_brayton(T1=TMIN, P1=PMIN, pressure_ratio=RP,  T3=TMAX, show_plot=False)
re = Ericsson.cycle_ericsson(T_min=TMIN, T_max=TMAX, P_min=PMIN, P_max=PMAX, show_plot=False)
rs = Stirling.cycle_stirling(T_min=TMIN, T_max=TMAX, V_min=VMIN, V_max=VMAX, show_plot=False)
print("Cycles calculés.")

LB = ["1→2 Isentropique", "2→3 Isobare",  "3→4 Isentropique", "4→1 Isobare"]
LE = ["1→2 Isotherme",    "2→3 Isobare",  "3→4 Isotherme",    "4→1 Isobare"]
LS = ["1→2 Isotherme",    "2→3 Isochore", "3→4 Isotherme",    "4→1 Isochore"]

SEGS_PV = [("V12","P12"),("V23","P23"),("V34","P34"),("V41","P41")]
SEGS_TS = [("s12","T12"),("s23","T23"),("s34","T34"),("s41","T41")]
T_CORNERS = [TMIN, TMIN, TMAX, TMAX]

def pv_ts(res, title, col, labels, prefix):
    cv = res["curves"]
    # — P-V —
    fig, ax = plt.subplots(figsize=(3.0, 2.1))
    for (vk,pk), lb, c in zip(SEGS_PV, labels, SEG_C):
        ax.plot(cv[vk], np.array(cv[pk])/1e5, color=c, label=lb)
    ax.scatter(res["V"], [p/1e5 for p in res["P"]], s=18, color=C_D, zorder=5)
    for i,(v,p) in enumerate(zip(res["V"], res["P"])):
        ax.annotate(str(i+1),(v,p/1e5), xytext=(3,3), textcoords="offset points",
                    fontsize=6, color=C_D)
    ax.set_xlabel("Volume spéc. (m³/kg)"); ax.set_ylabel("Pression (bar)")
    ax.set_title(f"Cycle {title} — P-V", color=col, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.8, fontsize=5.5)
    save(f"{prefix}_PV.pdf")

    # — T-s —
    fig, ax = plt.subplots(figsize=(3.0, 2.1))
    for (sk,tk), lb, c in zip(SEGS_TS, labels, SEG_C):
        ax.plot(cv[sk], cv[tk], color=c, label=lb)
    ax.scatter(res["s"], T_CORNERS, s=18, color=C_D, zorder=5)
    for i,(s,t) in enumerate(zip(res["s"], T_CORNERS)):
        ax.annotate(str(i+1),(s,t), xytext=(3,3), textcoords="offset points",
                    fontsize=6, color=C_D)
    ax.set_xlabel("Entropie (J/kg/K)"); ax.set_ylabel("Température (K)")
    ax.set_title(f"Cycle {title} — T-s", color=col, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.8, fontsize=5.5)
    save(f"{prefix}_Ts.pdf")

print("Diagrammes P-V / T-s...")
pv_ts(rb, "Brayton",  C_B, LB, "brayton")
pv_ts(re, "Ericsson", C_E, LE, "ericsson")
pv_ts(rs, "Stirling", C_S, LS, "stirling")

# ============================================================
# 2. Gaz parfait vs CoolProp : η(Tmax)
# ============================================================
print("Gaz parfait vs CoolProp...")
Tmaxs = np.linspace(600, 1300, 28)
gp, cp_e, cp_b = [], [], []
for Tm in Tmaxs:
    gp.append((1 - TMIN/Tm)*100)
    try:
        r = Ericsson.cycle_ericsson(T_min=TMIN,T_max=float(Tm),P_min=PMIN,P_max=PMAX,show_plot=False)
        cp_e.append(r["energetics"]["eta"]*100)
    except: cp_e.append(np.nan)
    try:
        r = Brayton.cycle_brayton(T1=TMIN,P1=PMIN,pressure_ratio=RP,T3=float(Tm),show_plot=False)
        cp_b.append(r["energetics"]["eta"]*100)
    except: cp_b.append(np.nan)

fig, ax = plt.subplots(figsize=(4.2, 2.8))
ax.plot(Tmaxs, gp,   color=C_A, ls="--", label="Ericsson gaz parfait")
ax.plot(Tmaxs, cp_e, color=C_E,          label="Ericsson CoolProp")
ax.plot(Tmaxs, cp_b, color=C_B,          label=f"Brayton CoolProp ($r_p$={RP})")
ax.set_xlabel("$T_{\\mathrm{max}}$ (K)"); ax.set_ylabel("Rendement (%)")
ax.set_title("Gaz parfait vs gaz réel (CoolProp)"); ax.legend()
save("gp_vs_coolprop.pdf")

# ============================================================
# 3. η(Tmax) — 3 cycles
# ============================================================
print("η vs Tmax...")
eta_e, eta_s, eta_b, carnot = [], [], [], []
for Tm in Tmaxs:
    carnot.append((1 - TMIN/Tm)*100)
    try:
        r = Ericsson.cycle_ericsson(T_min=TMIN,T_max=float(Tm),P_min=PMIN,P_max=PMAX,show_plot=False)
        eta_e.append(r["energetics"]["eta"]*100)
    except: eta_e.append(np.nan)
    try:
        r = Stirling.cycle_stirling(T_min=TMIN,T_max=float(Tm),V_min=VMIN,V_max=VMAX,show_plot=False)
        eta_s.append(r["energetics"]["eta"]*100)
    except: eta_s.append(np.nan)
    try:
        r = Brayton.cycle_brayton(T1=TMIN,P1=PMIN,pressure_ratio=RP,T3=float(Tm),show_plot=False)
        eta_b.append(r["energetics"]["eta"]*100)
    except: eta_b.append(np.nan)

fig, ax = plt.subplots(figsize=(4.5, 3.0))
ax.plot(Tmaxs, carnot, color=C_D, ls=":", lw=1.2, label="Carnot")
ax.plot(Tmaxs, eta_e,  color=C_E,              label="Ericsson")
ax.plot(Tmaxs, eta_s,  color=C_S, ls="--",     label="Stirling")
ax.plot(Tmaxs, eta_b,  color=C_B,              label=f"Brayton ($r_p$={RP})")
ax.set_xlabel("$T_{\\mathrm{max}}$ (K)"); ax.set_ylabel("Rendement (%)")
ax.set_title("Influence de $T_{\\mathrm{max}}$ sur $\\eta$"); ax.legend()
save("eta_Tmax.pdf")

# ============================================================
# 4. Wnet(Pmax) — Ericsson
# ============================================================
print("Wnet vs Pmax...")
Pmaxs = np.linspace(5e5, 30e5, 22)
wnet_e = []
for Pm in Pmaxs:
    try:
        r = Ericsson.cycle_ericsson(T_min=TMIN,T_max=TMAX,P_min=PMIN,P_max=float(Pm),show_plot=False)
        wnet_e.append(r["energetics"]["W_cycle"]/1000)
    except: wnet_e.append(np.nan)

fig, ax = plt.subplots(figsize=(3.8, 2.6))
ax.plot(Pmaxs/1e5, wnet_e, color=C_E)
ax.set_xlabel("$P_{\\mathrm{max}}$ (bar)"); ax.set_ylabel("$W_{\\mathrm{net}}$ (kJ/kg)")
ax.set_title("Ericsson : $W_{\\mathrm{net}}$ vs $P_{\\mathrm{max}}$")
save("Wnet_Pmax.pdf")

# ============================================================
# 5. η(rp) — Brayton vs Ericsson
# ============================================================
print("η vs rp...")
RPs = np.arange(2, 25)
eta_b_rp, eta_e_rp = [], []
for rp in RPs:
    try:
        r = Brayton.cycle_brayton(T1=TMIN,P1=PMIN,pressure_ratio=int(rp),T3=TMAX,show_plot=False)
        eta_b_rp.append(r["energetics"]["eta"]*100)
    except: eta_b_rp.append(np.nan)
    try:
        r = Ericsson.cycle_ericsson(T_min=TMIN,T_max=TMAX,P_min=PMIN,P_max=PMIN*rp,show_plot=False)
        eta_e_rp.append(r["energetics"]["eta"]*100)
    except: eta_e_rp.append(np.nan)

fig, ax = plt.subplots(figsize=(3.8, 2.6))
ax.plot(RPs, eta_b_rp, color=C_B, label="Brayton")
ax.plot(RPs, eta_e_rp, color=C_E, label="Ericsson (constant)")
ax.set_xlabel("Rapport de pression $r_p$"); ax.set_ylabel("Rendement (%)")
ax.set_title("$\\eta$ vs $r_p$ : Brayton / Ericsson"); ax.legend()
save("eta_rp.pdf")

# ============================================================
# 6. η(Tmin) — Brayton / Ericsson / Stirling
# ============================================================
print("η vs Tmin...")
Tmins = np.linspace(280, 450, 22)
eta_b_t, eta_e_t, eta_s_t, carnot_t = [], [], [], []
for Tm in Tmins:
    carnot_t.append((1 - Tm/TMAX)*100)
    try:
        r = Brayton.cycle_brayton(T1=float(Tm),P1=PMIN,pressure_ratio=RP,T3=TMAX,show_plot=False)
        eta_b_t.append(r["energetics"]["eta"]*100)
    except: eta_b_t.append(np.nan)
    try:
        r = Ericsson.cycle_ericsson(T_min=float(Tm),T_max=TMAX,P_min=PMIN,P_max=PMAX,show_plot=False)
        eta_e_t.append(r["energetics"]["eta"]*100)
    except: eta_e_t.append(np.nan)
    try:
        r = Stirling.cycle_stirling(T_min=float(Tm),T_max=TMAX,V_min=VMIN,V_max=VMAX,show_plot=False)
        eta_s_t.append(r["energetics"]["eta"]*100)
    except: eta_s_t.append(np.nan)

fig, ax = plt.subplots(figsize=(4.2, 2.8))
ax.plot(Tmins, carnot_t, color=C_D, ls=":",  lw=1.2, label="Carnot")
ax.plot(Tmins, eta_b_t,  color=C_B, ls="-.",          label="Brayton")
ax.plot(Tmins, eta_e_t,  color=C_E,                   label="Ericsson")
ax.plot(Tmins, eta_s_t,  color=C_S, ls="--",          label="Stirling")
ax.set_xlabel("$T_{\\mathrm{min}}$ (K)"); ax.set_ylabel("Rendement (%)")
ax.set_title("Influence de $T_{\\mathrm{min}}$ sur $\\eta$"); ax.legend()
save("eta_Tmin.pdf")

# ── 6b. Qin(Tmin) — Ericsson / Stirling ─────────────────────
print("Qin vs Tmin...")
qin_b_t, qin_e_t, qin_s_t = [], [], []
for Tm in Tmins:
    try:
        r = Brayton.cycle_brayton(T1=float(Tm),P1=PMIN,pressure_ratio=RP,T3=TMAX,show_plot=False)
        qin_b_t.append(r["energetics"]["Q_in"]/1000)
    except: qin_b_t.append(np.nan)
    try:
        r = Ericsson.cycle_ericsson(T_min=float(Tm),T_max=TMAX,P_min=PMIN,P_max=PMAX,show_plot=False)
        qin_e_t.append(r["energetics"]["Q_in"]/1000)
    except: qin_e_t.append(np.nan)
    try:
        r = Stirling.cycle_stirling(T_min=float(Tm),T_max=TMAX,V_min=VMIN,V_max=VMAX,show_plot=False)
        qin_s_t.append(r["energetics"]["Q_in"]/1000)
    except: qin_s_t.append(np.nan)

fig, ax = plt.subplots(figsize=(4.2, 2.8))
ax.plot(Tmins, qin_b_t, color=C_B, ls="-.",  label="Brayton")
ax.plot(Tmins, qin_e_t, color=C_E,            label="Ericsson")
ax.plot(Tmins, qin_s_t, color=C_S, ls="--",  label="Stirling")
ax.set_xlabel("$T_{\\mathrm{min}}$ (K)"); ax.set_ylabel("$Q_{\\mathrm{in}}$ (kJ/kg)")
ax.set_title("Influence de $T_{\\mathrm{min}}$ sur $Q_{\\mathrm{in}}$"); ax.legend()
save("Qin_Tmin.pdf")

# ============================================================
# 7. Comparaison η et Wnet — barres
# ============================================================
print("Comparaisons barres...")
cycles = ["Brayton", "Ericsson", "Stirling"]
colors = [C_B, C_E, C_S]
etas  = [rb["energetics"]["eta"]*100,
         re["energetics"]["eta"]*100,
         rs["energetics"]["eta"]*100]
wnets = [rb["energetics"]["W_net"]/1000,
         re["energetics"]["W_cycle"]/1000,
         rs["energetics"]["W_cycle"]/1000]

fig, ax = plt.subplots(figsize=(3.8, 2.6))
bars = ax.bar(cycles, etas, color=colors, width=0.5, edgecolor="white")
for b, v in zip(bars, etas):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.4,
            f"{v:.1f}%", ha="center", fontsize=7, fontweight="bold")
ax.set_ylabel("Rendement (%)"); ax.set_ylim(0, max(etas)*1.18)
ax.set_title("Comparaison $\\eta$ — 3 cycles\n($T_{\\min}$=300 K, $T_{\\max}$=1100 K)")
save("comparaison_eta.pdf")

fig, ax = plt.subplots(figsize=(3.8, 2.6))
bars = ax.bar(cycles, wnets, color=colors, width=0.5, edgecolor="white")
for b, v in zip(bars, wnets):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1.5,
            f"{v:.0f}", ha="center", fontsize=7, fontweight="bold")
ax.set_ylabel("$W_{\\mathrm{net}}$ (kJ/kg)"); ax.set_ylim(0, max(wnets)*1.20)
ax.set_title("Comparaison $W_{\\mathrm{net}}$ — 3 cycles")
save("comparaison_Wnet.pdf")

print("\nToutes les figures générées dans fig/")
