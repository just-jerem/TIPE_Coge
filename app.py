import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from src import Brayton, Ericsson

# ================================
# Titre de l'application
# ================================
st.title("Simulation de Cycles Thermodynamiques")

# ================================
# Choix du cycle via la sidebar
# ================================
choix_cycle = st.sidebar.selectbox(
    "Choisir le cycle thermodynamique",
    ("Cycle de Brayton", "Cycle d'Ericsson")
)

# ================================
# Paramètres du cycle
# ================================
st.sidebar.header("Paramètres du cycle")

if choix_cycle == "Cycle d'Ericsson":
    T_min = st.sidebar.slider("Température Minimale (K)", 250, 500, 300)
    T_max = st.sidebar.slider("Température Maximale (K)", 500, 1500, 1100)
    P_min = st.sidebar.slider("Pression Minimale (Pa)", 1e5, 5e5, 2e5)
    P_max = st.sidebar.slider("Pression Maximale (Pa)", 1e6, 30e5, 20e5)

    # Calcul du cycle
    results = Ericsson.cycle_ericsson(T_min=T_min, T_max=T_max, P_min=P_min, P_max=P_max, show_plot=False)

else:
    T1 = st.sidebar.slider("Température entrée compresseur (K)", 250, 500, 300)
    P1 = st.sidebar.slider("Pression entrée compresseur (Pa)", 1e5, 5e5, 2e5)
    pressure_ratio = st.sidebar.slider("Rapport de pression", 2, 20, 10)
    T3 = st.sidebar.slider("Température sortie chambre (K)", 800, 1500, 1100)

    # Calcul du cycle
    results = Brayton.cycle_brayton(T1=T1, P1=P1, pressure_ratio=pressure_ratio, T3=T3, show_plot=False)

# ================================
# Paramètres diagrammes PV et TS (ne relance pas le calcul)
# ================================
st.sidebar.header("Paramètres diagrammes")
pv_color = st.sidebar.color_picker("Couleur P-V", "#FF0000")
ts_color = st.sidebar.color_picker("Couleur T-S", "#0000FF")

# ================================
# Affichage des diagrammes P-V et T-S
# ================================
curves = results["curves"]

fig, ax = plt.subplots(1, 2, figsize=(12,5))

# P-V
ax[0].plot(curves["V12"], np.array(curves["P12"])/1e5, color=pv_color, label="1→2")
ax[0].plot(curves["V23"], np.array(curves["P23"])/1e5, color='green', label="2→3")
ax[0].plot(curves["V34"], np.array(curves["P34"])/1e5, color='orange', label="3→4")
ax[0].plot(curves["V41"], np.array(curves["P41"])/1e5, color='magenta', label="4→1")
ax[0].set_xlabel("Volume spécifique (m³/kg)")
ax[0].set_ylabel("Pression (bar)")
ax[0].set_title(f"{choix_cycle} (P-V)")
ax[0].grid()
ax[0].legend()

# T-S
ax[1].plot(curves["s12"], curves["T12"], color=ts_color, label="1→2")
ax[1].plot(curves["s23"], curves["T23"], color='green', label="2→3")
ax[1].plot(curves["s34"], curves["T34"], color='orange', label="3→4")
ax[1].plot(curves["s41"], curves["T41"], color='magenta', label="4→1")
ax[1].set_xlabel("Entropie massique (J/kg/K)")
ax[1].set_ylabel("Température (K)")
ax[1].set_title(f"{choix_cycle} (T-S)")
ax[1].grid()
ax[1].legend()

st.pyplot(fig)

# ================================
# Affichage des bilans énergétiques dans un tableau
# ================================
ener = results["energetics"]

if choix_cycle == "Cycle d'Ericsson":
    df = pd.DataFrame({
        "Grandeur": ["Chaleur reçue Q_in", "Chaleur rejetée Q_out", "Travail net W_cycle", "Rendement η"],
        "Valeur": [
            ener.get("Q_in",0)/1000,
            ener.get("Q_out",0)/1000,
            ener.get("W_cycle",0)/1000,
            ener.get("eta",0)*100
        ],
        "Unités": ["kJ/kg","kJ/kg","kJ/kg","%"]
    })
else:
    df = pd.DataFrame({
        "Grandeur": ["Travail compresseur W_comp", "Travail turbine W_turb", "Travail net W_net", "Chaleur reçue Q_in", "Rendement η"],
        "Valeur": [
            ener.get("W_comp",0)/1000,
            ener.get("W_turb",0)/1000,
            ener.get("W_net",0)/1000,
            ener.get("Q_in",0)/1000,
            ener.get("eta",0)*100
        ],
        "Unités": ["kJ/kg","kJ/kg","kJ/kg","kJ/kg","%"]
    })

st.subheader("Bilan énergétique")
st.dataframe(df)

# ================================
# Étude paramétrique
# ================================
st.markdown("---")
st.header(f"Étude paramétrique - {choix_cycle}")

st.sidebar.header("Paramètres étude paramétrique")

# Sliders pour paramétrer l'étude
if choix_cycle == "Cycle d'Ericsson":
    Tmin_min = st.sidebar.number_input("Tmin min (K)", 250, 500, 280)
    Tmin_max = st.sidebar.number_input("Tmin max (K)", 250, 500, 330)
    Tmin_points = st.sidebar.number_input("Nombre points Tmin", 2, 20, 6)

    Tmax_min = st.sidebar.number_input("Tmax min (K)", 500, 1500, 900)
    Tmax_max = st.sidebar.number_input("Tmax max (K)", 500, 1500, 1300)
    Tmax_points = st.sidebar.number_input("Nombre points Tmax", 2, 20, 6)

    Pmin_min = st.sidebar.number_input("Pmin min (Pa)", 1e5, 5e5, 1e5)
    Pmin_max = st.sidebar.number_input("Pmin max (Pa)", 1e5, 5e5, 3e5)
    Pmin_points = st.sidebar.number_input("Nombre points Pmin", 2, 10, 5)

    Pmax_min = st.sidebar.number_input("Pmax min (Pa)", 1e6, 30e5, 15e5)
    Pmax_max = st.sidebar.number_input("Pmax max (Pa)", 1e6, 30e5, 30e5)
    Pmax_points = st.sidebar.number_input("Nombre points Pmax", 2, 10, 5)

else:
    T1_min = st.sidebar.number_input("T1 min (K)", 250, 500, 280)
    T1_max = st.sidebar.number_input("T1 max (K)", 250, 500, 330)
    T1_points = st.sidebar.number_input("Nombre points T1", 2, 20, 6)

    T3_min = st.sidebar.number_input("T3 min (K)", 800, 1500, 900)
    T3_max = st.sidebar.number_input("T3 max (K)", 800, 1500, 1300)
    T3_points = st.sidebar.number_input("Nombre points T3", 2, 20, 6)

    P1_min = st.sidebar.number_input("P1 min (Pa)", 1e5, 5e5, 1e5)
    P1_max = st.sidebar.number_input("P1 max (Pa)", 1e5, 5e5, 3e5)
    P1_points = st.sidebar.number_input("Nombre points P1", 2, 10, 5)

    rp_min = st.sidebar.number_input("Rapport pression min", 2, 20, 2)
    rp_max = st.sidebar.number_input("Rapport pression max", 2, 20, 10)
    rp_points = st.sidebar.number_input("Nombre points Rp", 2, 10, 5)

run_study = st.button("Lancer l'étude paramétrique")

if run_study:
    st.info("Étude en cours...")

    progress_bar = st.progress(0)
    eta_list = []
    wnet_list = []
    qin_list = []
    combo_list = []

    if choix_cycle == "Cycle d'Ericsson":
        Tmin_values = np.linspace(Tmin_min, Tmin_max, Tmin_points)
        Tmax_values = np.linspace(Tmax_min, Tmax_max, Tmax_points)
        Pmin_values = np.linspace(Pmin_min, Pmin_max, Pmin_points)
        Pmax_values = np.linspace(Pmax_min, Pmax_max, Pmax_points)

        total = len(Tmin_values)*len(Tmax_values)*len(Pmin_values)*len(Pmax_values)
        count = 0

        fig_real, ax_real = plt.subplots()
        st.pyplot(fig_real)

        for Tmin in Tmin_values:
            for Tmax in Tmax_values:
                for Pmin in Pmin_values:
                    for Pmax in Pmax_values:
                        if Pmax <= Pmin or Tmax <= Tmin:
                            continue
                        res = Ericsson.cycle_ericsson(Tmin, Tmax, Pmin, Pmax, show_plot=False)
                        ener_tmp = res["energetics"]

                        eta_list.append(ener_tmp.get("eta",0))
                        wnet_list.append(ener_tmp.get("W_cycle",0))
                        qin_list.append(ener_tmp.get("Q_in",0))
                        combo_list.append((Tmin, Tmax, Pmin, Pmax))

                        # Tracé en temps réel
                        ax_real.clear()
                        ax_real.plot(eta_list, label="η")
                        ax_real.plot(np.array(wnet_list)/1000, label="Wnet (kJ/kg)")
                        ax_real.plot(np.array(qin_list)/1000, label="Qin (kJ/kg)")
                        ax_real.set_xlabel("Configuration")
                        ax_real.set_ylabel("Valeur")
                        ax_real.grid()
                        ax_real.legend()
                        st.pyplot(fig_real)

                        count += 1
                        progress_bar.progress(count/total)

    else:
        T1_values = np.linspace(T1_min, T1_max, T1_points)
        T3_values = np.linspace(T3_min, T3_max, T3_points)
        P1_values = np.linspace(P1_min, P1_max, P1_points)
        rp_values = np.linspace(rp_min, rp_max, rp_points)

        total = len(T1_values)*len(T3_values)*len(P1_values)*len(rp_values)
        count = 0

        fig_real, ax_real = plt.subplots()
        st.pyplot(fig_real)

        for T1 in T1_values:
            for T3 in T3_values:
                for P1 in P1_values:
                    for rp in rp_values:
                        res = Brayton.cycle_brayton(T1, P1, rp, T3, show_plot=False)
                        ener_tmp = res["energetics"]

                        eta_list.append(ener_tmp.get("eta",0))
                        wnet_list.append(ener_tmp.get("W_net",0))
                        qin_list.append(ener_tmp.get("Q_in",0))
                        combo_list.append((T1, T3, P1, rp))

                        # Tracé en temps réel
                        ax_real.clear()
                        ax_real.plot(eta_list, label="η")
                        ax_real.plot(np.array(wnet_list)/1000, label="Wnet (kJ/kg)")
                        ax_real.plot(np.array(qin_list)/1000, label="Qin (kJ/kg)")
                        ax_real.set_xlabel("Configuration")
                        ax_real.set_ylabel("Valeur")
                        ax_real.grid()
                        ax_real.legend()
                        st.pyplot(fig_real)

                        count += 1
                        progress_bar.progress(count/total)

    st.success("✅ Étude terminée")

    # Meilleure configuration
    index_max_eta = np.argmax(eta_list)
    best_case = combo_list[index_max_eta]

    st.subheader("Meilleure configuration (rendement max)")
    st.write(best_case)
    st.write(f"Rendement max : {eta_list[index_max_eta]*100:.2f} %")
