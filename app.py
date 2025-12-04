import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from src import Brayton, Ericsson
import time

st.title("Simulation de Cycles Thermodynamiques")

# ================================
# Choix du cycle
# ================================
choix_cycle = st.selectbox("Choisir le cycle thermodynamique", ("Cycle de Brayton", "Cycle d'Ericsson"))

# ================================
# Paramètres diagrammes PV/TS
# ================================
with st.expander("Paramètres diagrammes P-V et T-S", expanded=True):
    if choix_cycle == "Cycle d'Ericsson":
        T_min = st.slider("Température Minimale (K)", 250, 500, 300)
        T_max = st.slider("Température Maximale (K)", 500, 1500, 1100)
        P_min = st.slider("Pression Minimale (Pa)", 1e5, 5e5, 2e5)
        P_max = st.slider("Pression Maximale (Pa)", 1e6, 30e5, 20e5)
    else:
        T1 = st.slider("Température entrée compresseur (K)", 250, 500, 300)
        P1 = st.slider("Pression entrée compresseur (Pa)", 1e5, 5e5, 2e5)
        pressure_ratio = st.slider("Rapport de pression", 2, 20, 10)
        T3 = st.slider("Température sortie chambre (K)", 800, 1500, 1100)

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
        "Grandeur": ["Qin (kJ/kg)", "Qout (kJ/kg)", "W_cycle (kJ/kg)", "Rendement (%)"],
        "Valeur": [
            ener.get("Q_in",0)/1000,
            ener.get("Q_out",0)/1000,
            ener.get("W_cycle",0)/1000,
            ener.get("eta",0)*100
        ]
    })
else:  # Brayton
    df_ener = pd.DataFrame({
        "Grandeur": ["Wcomp (kJ/kg)", "Wturb (kJ/kg)", "Wnet (kJ/kg)", "Qin (kJ/kg)", "Rendement (%)"],
        "Valeur": [
            ener.get("W_comp",0)/1000,
            ener.get("W_turb",0)/1000,
            ener.get("W_net",0)/1000,
            ener.get("Q_in",0)/1000,
            ener.get("eta",0)*100
        ]
    })
st.table(df_ener)

# ================================
# Paramètres étude paramétrique
# ================================
with st.expander("Paramètres étude paramétrique", expanded=True):
    if choix_cycle == "Cycle d'Ericsson":
        Tmin_min = st.number_input("Tmin min (K)", 200, 500, 280)
        Tmin_max = st.number_input("Tmin max (K)", 200, 500, 330)
        Tmin_points = st.number_input("Nombre points Tmin", 2, 20, 6)
        Tmax_min = st.number_input("Tmax min (K)", 500, 1500, 900)
        Tmax_max = st.number_input("Tmax max (K)", 500, 1500, 1300)
        Tmax_points = st.number_input("Nombre points Tmax", 2, 20, 6)
        Pmin_min = st.number_input("Pmin min (Pa)", 1e5, 5e5, 1e5)
        Pmin_max = st.number_input("Pmin max (Pa)", 1e5, 5e5, 3e5)
        Pmin_points = st.number_input("Nombre points Pmin", 2, 10, 5)
        Pmax_min = st.number_input("Pmax min (Pa)", 1e6, 30e5, 15e5)
        Pmax_max = st.number_input("Pmax max (Pa)", 1e6, 30e5, 30e5)
        Pmax_points = st.number_input("Nombre points Pmax", 2, 10, 5)
    else:
        T1_min = st.number_input("T1 min (K)", 200, 500, 280)
        T1_max = st.number_input("T1 max (K)", 200, 500, 330)
        T1_points = st.number_input("Nombre points T1", 2, 20, 6)
        P1_min = st.number_input("P1 min (Pa)", 1e5, 5e5, 1e5)
        P1_max = st.number_input("P1 max (Pa)", 1e5, 5e5, 3e5)
        P1_points = st.number_input("Nombre points P1", 2, 10, 5)
        PR_min = st.number_input("PR min", 2, 20, 2)
        PR_max = st.number_input("PR max", 2, 20, 12)
        PR_points = st.number_input("Nombre points PR", 2, 10, 6)
        T3_min = st.number_input("T3 min (K)", 800, 1500, 900)
        T3_max = st.number_input("T3 max (K)", 800, 1500, 1300)
        T3_points = st.number_input("Nombre points T3", 2, 20, 6)

# ================================
# Lancer étude paramétrique
# ================================
run_study = st.button("Lancer l'étude paramétrique")

if run_study:
    st.info("Simulation en cours...")
    progress_bar = st.progress(0)
    timer_placeholder = st.empty()
    table_placeholder = st.empty()
    chart_q = st.empty()
    chart_w = st.empty()
    chart_eta = st.empty()

    results_list = []
    k = 0
    start_time = time.time()

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
                        chart_q.line_chart(df[["Qin (kJ/kg)","Qout (kJ/kg)"]])
                        chart_w.line_chart(df[["Wnet (kJ/kg)"]])
                        chart_eta.line_chart(df[["Rendement (%)"]])
                        k += 1
                        progress_bar.progress(int(100*k/total))
                        elapsed = time.time() - start_time
                        remaining = (elapsed/k)*(total-k) if k>0 else 0
                        timer_placeholder.info(f"Temps restant estimé : {int(remaining)} s")
    else:  # Brayton
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
                            "Wcomp (kJ/kg)": ener.get("W_comp",0)/1000,
                            "Wturb (kJ/kg)": ener.get("W_turb",0)/1000,
                            "Wnet (kJ/kg)": ener.get("W_net",0)/1000,
                            "Qin (kJ/kg)": ener.get("Q_in",0)/1000,
                            "Rendement (%)": ener.get("eta",0)*100
                        })
                        df = pd.DataFrame(results_list)
                        table_placeholder.dataframe(df)
                        chart_q.line_chart(df[["Wcomp (kJ/kg)","Wturb (kJ/kg)"]])
                        chart_w.line_chart(df[["Wnet (kJ/kg)"]])
                        chart_eta.line_chart(df[["Rendement (%)"]])
                        k += 1
                        progress_bar.progress(int(100*k/total))
                        elapsed = time.time() - start_time
                        remaining = (elapsed/k)*(total-k) if k>0 else 0
                        timer_placeholder.info(f"Temps restant estimé : {int(remaining)} s")

    st.success("Étude paramétrique terminée !")
    # Meilleure configuration
    idx_max = df["Rendement (%)"].idxmax()
    best = df.loc[idx_max]
    st.subheader("Meilleure configuration")
    st.table(best)

# ================================
# Import CSV
# ================================
st.markdown("---")
st.header("Importer un CSV pour tracer les résultats existants")

csv_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])

if csv_file is not None:
    df_csv = pd.read_csv(csv_file)
    st.subheader("Aperçu des données importées")
    st.dataframe(df_csv.head())

    # Détection du type de cycle
    if all(col in df_csv.columns for col in ["Qin (kJ/kg)","Qout (kJ/kg)","Wnet (kJ/kg)","Rendement (%)"]):
        cycle_csv = "Ericsson"
        st.info("Cycle détecté : Ericsson")
    elif all(col in df_csv.columns for col in ["Wcomp (kJ/kg)","Wturb (kJ/kg)","Wnet (kJ/kg)","Qin (kJ/kg)","Rendement (%)"]):
        cycle_csv = "Brayton"
        st.info("Cycle détecté : Brayton")
    else:
        st.warning("Impossible de détecter le type de cycle automatiquement. Vérifiez les colonnes.")
        cycle_csv = None

    if cycle_csv is not None:
        # Tracer graphiques
        st.subheader("Graphiques à partir du CSV")
        if cycle_csv == "Ericsson":
            st.line_chart(df_csv[["Qin (kJ/kg)","Qout (kJ/kg)"]], height=200)
            st.line_chart(df_csv[["Wnet (kJ/kg)"]], height=200)
            st.line_chart(df_csv[["Rendement (%)"]], height=200)
        else:  # Brayton
            st.line_chart(df_csv[["Wcomp (kJ/kg)","Wturb (kJ/kg)"]], height=200)
            st.line_chart(df_csv[["Wnet (kJ/kg)"]], height=200)
            st.line_chart(df_csv[["Rendement (%)"]], height=200)

        # Meilleure configuration
        idx_max = df_csv["Rendement (%)"].idxmax()
        best_csv = df_csv.loc[idx_max]
        st.subheader("Meilleure configuration")
        st.table(best_csv)


