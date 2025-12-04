import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from src import Brayton, Ericsson
import time  # <-- Pour le timer

# ================================
# Titre
# ================================
st.title("Simulation de Cycles Thermodynamiques")

# ================================
# Choix du cycle
# ================================
choix_cycle = st.sidebar.selectbox(
    "Choisir le cycle thermodynamique",
    ("Cycle de Brayton", "Cycle d'Ericsson")
)

# ================================
# Paramètres diagrammes (st.form pour ne pas rerun)
# ================================
st.sidebar.markdown("### Paramètres diagrammes P-V et T-S")
with st.form("form_diagrams", clear_on_submit=False):
    if choix_cycle == "Cycle d'Ericsson":
        T_min = st.slider("Température Minimale (K)", 250, 500, 300, key="Tmin_diagram")
        T_max = st.slider("Température Maximale (K)", 500, 1500, 1100, key="Tmax_diagram")
        P_min = st.slider("Pression Minimale (Pa)", 1e5, 5e5, 2e5, key="Pmin_diagram")
        P_max = st.slider("Pression Maximale (Pa)", 1e6, 30e5, 20e5, key="Pmax_diagram")
    else:
        T1 = st.slider("Température entrée compresseur (K)", 250, 500, 300, key="T1_diagram")
        P1 = st.slider("Pression entrée compresseur (Pa)", 1e5, 5e5, 2e5, key="P1_diagram")
        pressure_ratio = st.slider("Rapport de pression", 2, 20, 10, key="PR_diagram")
        T3 = st.slider("Température sortie chambre (K)", 800, 1500, 1100, key="T3_diagram")
    st.form_submit_button("Mettre à jour diagrammes")

# ================================
# Calcul du cycle principal
# ================================
if choix_cycle == "Cycle d'Ericsson":
    results = Ericsson.cycle_ericsson(T_min=T_min, T_max=T_max, P_min=P_min, P_max=P_max, show_plot=False)
else:
    results = Brayton.cycle_brayton(T1=T1, P1=P1, pressure_ratio=pressure_ratio, T3=T3, show_plot=False)

# ================================
# Diagrammes P-V et T-S
# ================================
curves = results["curves"]

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(curves["V12"], np.array(curves["P12"])/1e5, 'r', label="1→2")
plt.plot(curves["V23"], np.array(curves["P23"])/1e5, 'b', label="2→3")
plt.plot(curves["V34"], np.array(curves["P34"])/1e5, 'g', label="3→4")
plt.plot(curves["V41"], np.array(curves["P41"])/1e5, 'm', label="4→1")
plt.xlabel("Volume spécifique (m³/kg)")
plt.ylabel("Pression (bar)")
plt.title(f"{choix_cycle} (P-V)")
plt.grid()
plt.legend()

plt.subplot(1,2,2)
plt.plot(curves["s12"], curves["T12"], 'r', label="1→2")
plt.plot(curves["s23"], curves["T23"], 'b', label="2→3")
plt.plot(curves["s34"], curves["T34"], 'g', label="3→4")
plt.plot(curves["s41"], curves["T41"], 'm', label="4→1")
plt.xlabel("Entropie massique (J/kg/K)")
plt.ylabel("Température (K)")
plt.title(f"{choix_cycle} (T-S)")
plt.grid()
plt.legend()
st.pyplot(plt)

# ================================
# Tableau bilans énergétiques
# ================================
ener = results["energetics"]
st.subheader("Bilan énergétique du cycle")
if choix_cycle == "Cycle d'Ericsson":
    df_ener = pd.DataFrame({
        "Grandeur": ["Chaleur reçue (Qin)", "Chaleur rejetée (Qout)", "Travail net (Wnet)", "Rendement (%)"],
        "Valeur": [
            ener.get("Q_in", 0)/1000,
            ener.get("Q_out", 0)/1000,
            ener.get("W_cycle", 0)/1000,
            ener.get("eta", 0)*100
        ]
    })
else:
    df_ener = pd.DataFrame({
        "Grandeur": ["Chaleur reçue (Qin)", "Travail compresseur (Wcomp)", "Travail turbine (Wturb)", "Travail net (Wnet)", "Rendement (%)"],
        "Valeur": [
            ener.get("Q_in", 0)/1000,
            ener.get("W_comp", 0)/1000,
            ener.get("W_turb", 0)/1000,
            ener.get("W_net", 0)/1000,
            ener.get("eta", 0)*100
        ]
    })
st.table(df_ener)

# ================================
# Étude paramétrique
# ================================
st.markdown("---")
st.header("Étude paramétrique")
st.sidebar.markdown("### Paramètres étude paramétrique")

# Paramètres modifiables
if choix_cycle == "Cycle d'Ericsson":
    Tmin_min = st.sidebar.number_input("Tmin min (K)", 200, 500, 280)
    Tmin_max = st.sidebar.number_input("Tmin max (K)", 200, 500, 330)
    Tmin_points = st.sidebar.number_input("Nombre de points Tmin", 2, 20, 6)
    Tmax_min = st.sidebar.number_input("Tmax min (K)", 500, 1500, 900)
    Tmax_max = st.sidebar.number_input("Tmax max (K)", 500, 1500, 1300)
    Tmax_points = st.sidebar.number_input("Nombre de points Tmax", 2, 20, 6)
    Pmin_min = st.sidebar.number_input("Pmin min (Pa)", 1e5, 5e5, 1e5)
    Pmin_max = st.sidebar.number_input("Pmin max (Pa)", 1e5, 5e5, 3e5)
    Pmin_points = st.sidebar.number_input("Nombre de points Pmin", 2, 10, 5)
    Pmax_min = st.sidebar.number_input("Pmax min (Pa)", 1e6, 30e5, 15e5)
    Pmax_max = st.sidebar.number_input("Pmax max (Pa)", 1e6, 30e5, 30e5)
    Pmax_points = st.sidebar.number_input("Nombre de points Pmax", 2, 10, 5)
else:
    T1_min = st.sidebar.number_input("T1 min (K)", 200, 500, 280)
    T1_max = st.sidebar.number_input("T1 max (K)", 200, 500, 330)
    T1_points = st.sidebar.number_input("Nombre de points T1", 2, 20, 6)
    P1_min = st.sidebar.number_input("P1 min (Pa)", 1e5, 5e5, 1e5)
    P1_max = st.sidebar.number_input("P1 max (Pa)", 1e5, 5e5, 3e5)
    P1_points = st.sidebar.number_input("Nombre de points P1", 2, 10, 5)
    PR_min = st.sidebar.number_input("PR min", 2, 20, 2)
    PR_max = st.sidebar.number_input("PR max", 2, 20, 12)
    PR_points = st.sidebar.number_input("Nombre de points PR", 2, 10, 6)
    T3_min = st.sidebar.number_input("T3 min (K)", 800, 1500, 900)
    T3_max = st.sidebar.number_input("T3 max (K)", 800, 1500, 1300)
    T3_points = st.sidebar.number_input("Nombre de points T3", 2, 20, 6)

# Lancer étude
run_study = st.button("Lancer étude paramétrique")

if run_study:
    st.info("Simulation en cours...")
    progress_bar = st.progress(0)
    chart_placeholder = st.empty()
    table_placeholder = st.empty()
    timer_placeholder = st.empty()
    results_list = []
    k = 0
    start_time = time.time()  # <--- timer

    if choix_cycle == "Cycle d'Ericsson":
        Tmin_values = np.linspace(Tmin_min, Tmin_max, Tmin_points)
        Tmax_values = np.linspace(Tmax_min, Tmax_max, Tmax_points)
        Pmin_values = np.linspace(Pmin_min, Pmin_max, Pmin_points)
        Pmax_values = np.linspace(Pmax_min, Pmax_max, Pmax_points)
        total = len(Tmin_values)*len(Tmax_values)*len(Pmin_values)*len(Pmax_values)

        for Tmin in Tmin_values:
            for Tmax in Tmax_values:
                for Pmin in Pmin_values:
                    for Pmax in Pmax_values:
                        if Tmax <= Tmin or Pmax <= Pmin:
                            continue
                        res = Ericsson.cycle_ericsson(Tmin, Tmax, Pmin, Pmax, show_plot=False)
                        ener = res["energetics"]
                        results_list.append({
                            "Tmin": Tmin, "Tmax": Tmax,
                            "Pmin (bar)": Pmin/1e5, "Pmax (bar)": Pmax/1e5,
                            "Qin (kJ/kg)": ener.get("Q_in",0)/1000,
                            "Qout (kJ/kg)": ener.get("Q_out",0)/1000,
                            "Wnet (kJ/kg)": ener.get("W_cycle",0)/1000,
                            "Rendement (%)": ener.get("eta",0)*100
                        })
                        df = pd.DataFrame(results_list)
                        table_placeholder.dataframe(df)
                        chart_placeholder.line_chart(df[["Rendement (%)","Wnet (kJ/kg)","Qin (kJ/kg)"]])

                        # Timer estimé
                        elapsed = time.time() - start_time
                        remaining = elapsed / (k+1) * (total - (k+1))
                        timer_placeholder.text(f"Temps estimé restant : {int(remaining)} s")

                        k += 1
                        progress_bar.progress(int(100*k/total))
    else:
        T1_values = np.linspace(T1_min, T1_max, T1_points)
        P1_values = np.linspace(P1_min, P1_max, P1_points)
        PR_values = np.linspace(PR_min, PR_max, PR_points)
        T3_values = np.linspace(T3_min, T3_max, T3_points)
        total = len(T1_values)*len(P1_values)*len(PR_values)*len(T3_values)

        for T1 in T1_values:
            for P1 in P1_values:
                for PR in PR_values:
                    for T3 in T3_values:
                        res = Brayton.cycle_brayton(T1, P1, PR, T3, show_plot=False)
                        ener = res["energetics"]
                        results_list.append({
                            "T1": T1, "P1 (bar)": P1/1e5, "PR": PR, "T3": T3,
                            "Qin (kJ/kg)": ener.get("Q_in",0)/1000,
                            "Wcomp (kJ/kg)": ener.get("W_comp",0)/1000,
                            "Wturb (kJ/kg)": ener.get("W_turb",0)/1000,
                            "Wnet (kJ/kg)": ener.get("W_net",0)/1000,
                            "Rendement (%)": ener.get("eta",0)*100
                        })
                        df = pd.DataFrame(results_list)
                        table_placeholder.dataframe(df)
                        chart_placeholder.line_chart(df[["Rendement (%)","Wnet (kJ/kg)","Qin (kJ/kg)"]])

                        # Timer estimé
                        elapsed = time.time() - start_time
                        remaining = elapsed / (k+1) * (total - (k+1))
                        timer_placeholder.text(f"Temps estimé restant : {int(remaining)} s")

                        k += 1
                        progress_bar.progress(int(100*k/total))

    st.success("Étude paramétrique terminée !")

    # Meilleure configuration
    idx_max = df["Rendement (%)"].idxmax()
    best = df.loc[idx_max]
    st.subheader("Meilleure configuration")
    st.table(best)
