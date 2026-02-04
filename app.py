import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import altair as alt
from src import Brayton, Ericsson, Stirling
import time

# ========================================
# FONCTION HELPER POUR GRAPHIQUES ALTAIR
# ========================================
def create_overlay_chart(all_results, metric_col, metric_label, unit_labels, y_title):
    """
    Crée un graphique Altair avec tooltips améliorés pour afficher
    les valeurs des paramètres au survol.
    
    Args:
        all_results: dict {param_name: DataFrame}
        metric_col: nom de la colonne à tracer (ex: "Rendement (%)")
        metric_label: label pour l'axe Y
        unit_labels: dict des unités par paramètre
        y_title: titre de l'axe Y
    
    Returns:
        graphique Altair
    """
    # Construire un DataFrame long (format tidy) pour Altair
    dfs = []
    for param_name, df in all_results.items():
        df_copy = df.copy().reset_index(drop=True)
        df_copy["Point"] = df_copy.index
        
        # Calculer le pourcentage de progression dans la plage
        param_min = df_copy[param_name].min()
        param_max = df_copy[param_name].max()
        df_copy["Progression (%)"] = 100 * (df_copy[param_name] - param_min) / (param_max - param_min + 1e-10)
        
        # Formater le label du paramètre
        unit = unit_labels.get(param_name, "")
        if "P_" in param_name or param_name == "P1":
            label = f"{param_name} ({param_min/1e5:.1f} à {param_max/1e5:.1f} {unit})"
            # Convertir en bar pour affichage
            df_copy["Valeur affichée"] = df_copy[param_name] / 1e5
            df_copy["Unité"] = "bar"
        elif "V_" in param_name:
            label = f"{param_name} ({param_min:.2f} à {param_max:.2f} {unit})"
            df_copy["Valeur affichée"] = df_copy[param_name]
            df_copy["Unité"] = unit
        elif param_name == "Ratio de pression":
            label = f"Ratio ({int(param_min)} à {int(param_max)})"
            df_copy["Valeur affichée"] = df_copy[param_name]
            df_copy["Unité"] = ""
        else:
            label = f"{param_name} ({param_min:.0f} à {param_max:.0f} {unit})"
            df_copy["Valeur affichée"] = df_copy[param_name]
            df_copy["Unité"] = unit
        
        df_copy["Paramètre"] = label
        df_copy["Nom paramètre"] = param_name
        df_copy["Valeur paramètre"] = df_copy[param_name]
        df_copy["Métrique"] = df_copy[metric_col]
        dfs.append(df_copy[["Point", "Progression (%)", "Paramètre", "Nom paramètre", "Valeur paramètre", "Valeur affichée", "Unité", "Métrique"]])
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Sélection au survol - montre tous les points à la même position X
    nearest = alt.selection_point(
        nearest=True, 
        on="pointerover",
        fields=["Point"],
        empty=False
    )
    
    # Ligne de base avec tooltip amélioré
    line = alt.Chart(combined).mark_line(strokeWidth=2.5).encode(
        x=alt.X("Point:Q", title="Point de simulation"),
        y=alt.Y("Métrique:Q", title=y_title),
        color=alt.Color("Paramètre:N", title="Paramètre varié", 
                       legend=alt.Legend(orient="bottom", columns=2, labelLimit=300)),
        detail="Paramètre:N"
    )
    
    # Points pour le survol avec tooltip détaillé
    points = line.mark_circle(size=120, filled=True).encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip("Paramètre:N", title="Paramètre"),
            alt.Tooltip("Valeur affichée:Q", title="Valeur", format=".2f"),
            alt.Tooltip("Unité:N", title="Unité"),
            alt.Tooltip("Métrique:Q", title=f"{metric_label}", format=".2f"),
            alt.Tooltip("Progression (%):Q", title="Progression", format=".1f")
        ]
    ).add_params(nearest)
    
    # Ligne verticale au survol
    rules = alt.Chart(combined).mark_rule(color="gray", strokeDash=[4,4], strokeWidth=1).encode(
        x="Point:Q"
    ).transform_filter(nearest)
    
    # Combiner tous les layers
    chart = alt.layer(line, points, rules).properties(
        height=350
    ).interactive()
    
    return chart

st.title("Simulation de Cycles Thermodynamiques")

# ================================
# Choix du cycle
# ================================
choix_cycle = st.selectbox("Choisir le cycle thermodynamique", ("Cycle de Brayton", "Cycle d'Ericsson", "Cycle de Stirling"))

# ================================
# Paramètres diagrammes PV/TS
# ================================
with st.expander("Paramètres diagrammes P-V et T-S", expanded=True):
    if choix_cycle == "Cycle d'Ericsson":
        T_min = st.slider("Température Minimale (K)", 250, 500, 300)
        T_max = st.slider("Température Maximale (K)", 500, 1500, 1100)
        P_min = st.slider("Pression Minimale (Pa)", 1e5, 5e5, 2e5)
        P_max = st.slider("Pression Maximale (Pa)", 1e6, 30e5, 20e5)
    elif choix_cycle == "Cycle de Stirling":
        T_min_st = st.slider("Température Minimale (K)", 250, 500, 300, key="T_min_stirling")
        T_max_st = st.slider("Température Maximale (K)", 500, 1500, 1100, key="T_max_stirling")
        V_min_st = st.slider("Volume spécifique minimal (m³/kg)", 0.1, 0.8, 0.3, step=0.05, key="V_min_stirling")
        V_max_st = st.slider("Volume spécifique maximal (m³/kg)", 0.5, 2.0, 1.0, step=0.05, key="V_max_stirling")
    else:  # Brayton
        T1 = st.slider("Température entrée compresseur (K)", 250, 500, 300)
        P1 = st.slider("Pression entrée compresseur (Pa)", 1e5, 5e5, 2e5)
        pressure_ratio = st.slider("Rapport de pression", 2, 20, 10)
        T3 = st.slider("Température sortie chambre (K)", 800, 1500, 1100)

# ================================
# Calcul du cycle principal
# ================================
if choix_cycle == "Cycle d'Ericsson":
    results = Ericsson.cycle_ericsson(T_min=T_min, T_max=T_max, P_min=P_min, P_max=P_max, show_plot=False)
elif choix_cycle == "Cycle de Stirling":
    results = Stirling.cycle_stirling(T_min=T_min_st, T_max=T_max_st, V_min=V_min_st, V_max=V_max_st, show_plot=False)
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
elif choix_cycle == "Cycle de Stirling":
    df_ener = pd.DataFrame({
        "Grandeur": ["Qin (kJ/kg)", "Qout (kJ/kg)", "W_cycle (kJ/kg)", "Q_regen (kJ/kg)", "Rendement (%)", "Rendement Carnot (%)"],
        "Valeur": [
            ener.get("Q_in",0)/1000,
            ener.get("Q_out",0)/1000,
            ener.get("W_cycle",0)/1000,
            ener.get("Q_regen",0)/1000,
            ener.get("eta",0)*100,
            ener.get("eta_carnot",0)*100
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
    # Sélection des paramètres à faire varier
    if choix_cycle == "Cycle d'Ericsson":
        param_options = ["T_min", "T_max", "P_min", "P_max"]
        params_to_vary = st.multiselect("Quels paramètres faire varier ? (un à la fois, superposés)", param_options, default=["T_min"])
        
        # Paramètres fixes
        st.write("**Paramètres fixes :**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            T_min_fixed = st.number_input("T_min (K)", 250, 500, 300, key="T_min_fixed")
        with col2:
            T_max_fixed = st.number_input("T_max (K)", 500, 1500, 1100, key="T_max_fixed")
        with col3:
            P_min_fixed = st.number_input("P_min (Pa)", 1e5, 5e5, 2e5, key="P_min_fixed")
        with col4:
            P_max_fixed = st.number_input("P_max (Pa)", 1e6, 30e5, 20e5, key="P_max_fixed")
        
        # Paramètres à varier
        st.write("**Plages de variation :**")
        param_ranges = {}
        cols_ranges = st.columns(len(params_to_vary) if params_to_vary else 1)
        for idx, param in enumerate(params_to_vary):
            with cols_ranges[idx]:
                if param == "T_min":
                    param_ranges[param] = (
                        st.number_input(f"{param} min", 200, 400, 250, key=f"{param}_min"),
                        st.number_input(f"{param} max", 300, 500, 350, key=f"{param}_max")
                    )
                elif param == "T_max":
                    param_ranges[param] = (
                        st.number_input(f"{param} min", 600, 1500, 900, key=f"{param}_min"),
                        st.number_input(f"{param} max", 800, 1500, 1300, key=f"{param}_max")
                    )
                elif param == "P_min":
                    param_ranges[param] = (
                        st.number_input(f"{param} min", 1e5, 4e5, 1e5, key=f"{param}_min"),
                        st.number_input(f"{param} max", 2e5, 5e5, 3e5, key=f"{param}_max")
                    )
                else:  # P_max
                    param_ranges[param] = (
                        st.number_input(f"{param} min", 1e6, 25e5, 10e5, key=f"{param}_min"),
                        st.number_input(f"{param} max", 2e6, 30e5, 25e5, key=f"{param}_max")
                    )
        
        param_points = st.slider("Nombre de points par paramètre", 5, 50, 15, key="param_points_ericsson")
    
    elif choix_cycle == "Cycle de Stirling":
        # ----------------------------------------
        # PARAMETRES STIRLING
        # ----------------------------------------
        param_options = ["T_min", "T_max", "V_min", "V_max"]
        params_to_vary = st.multiselect("Quels paramètres faire varier ? (un à la fois, superposés)", param_options, default=["T_min"], key="params_stirling")
        
        # Paramètres fixes
        st.write("**Paramètres fixes :**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            T_min_fixed_st = st.number_input("T_min (K)", 250, 500, 300, key="T_min_fixed_st")
        with col2:
            T_max_fixed_st = st.number_input("T_max (K)", 500, 1500, 1100, key="T_max_fixed_st")
        with col3:
            V_min_fixed_st = st.number_input("V_min (m³/kg)", 0.1, 0.8, 0.3, step=0.05, key="V_min_fixed_st")
        with col4:
            V_max_fixed_st = st.number_input("V_max (m³/kg)", 0.5, 2.0, 1.0, step=0.05, key="V_max_fixed_st")
        
        # Paramètres à varier
        st.write("**Plages de variation :**")
        param_ranges = {}
        cols_ranges = st.columns(len(params_to_vary) if params_to_vary else 1)
        for idx, param in enumerate(params_to_vary):
            with cols_ranges[idx]:
                if param == "T_min":
                    param_ranges[param] = (
                        st.number_input(f"{param} min", 200, 400, 250, key=f"{param}_min_st"),
                        st.number_input(f"{param} max", 300, 500, 350, key=f"{param}_max_st")
                    )
                elif param == "T_max":
                    param_ranges[param] = (
                        st.number_input(f"{param} min", 600, 1500, 900, key=f"{param}_min_st"),
                        st.number_input(f"{param} max", 800, 1500, 1300, key=f"{param}_max_st")
                    )
                elif param == "V_min":
                    param_ranges[param] = (
                        st.number_input(f"{param} min", 0.1, 0.5, 0.2, step=0.05, key=f"{param}_min_st"),
                        st.number_input(f"{param} max", 0.3, 0.8, 0.5, step=0.05, key=f"{param}_max_st")
                    )
                else:  # V_max
                    param_ranges[param] = (
                        st.number_input(f"{param} min", 0.5, 1.5, 0.8, step=0.05, key=f"{param}_min_st"),
                        st.number_input(f"{param} max", 1.0, 2.0, 1.5, step=0.05, key=f"{param}_max_st")
                    )
        
        param_points = st.slider("Nombre de points par paramètre", 5, 50, 15, key="param_points_stirling")
    
    else:  # Brayton
        param_options = ["T1", "P1", "Ratio de pression", "T3"]
        params_to_vary = st.multiselect("Quels paramètres faire varier ? (un à la fois, superposés)", param_options, default=["T1"])
        
        # Paramètres fixes
        st.write("**Paramètres fixes :**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            T1_fixed = st.number_input("T1 (K)", 250, 500, 300, key="T1_fixed")
        with col2:
            P1_fixed = st.number_input("P1 (Pa)", 1e5, 5e5, 2e5, key="P1_fixed")
        with col3:
            PR_fixed = st.slider("Ratio de pression", 2, 20, 10, key="PR_fixed")
        with col4:
            T3_fixed = st.number_input("T3 (K)", 800, 1500, 1100, key="T3_fixed")
        
        # Paramètres à varier
        st.write("**Plages de variation :**")
        param_ranges = {}
        cols_ranges = st.columns(len(params_to_vary) if params_to_vary else 1)
        for idx, param in enumerate(params_to_vary):
            with cols_ranges[idx]:
                if param == "T1":
                    param_ranges[param] = (
                        st.number_input(f"{param} min", 250, 400, 260, help="Min: 250K", key=f"{param}_min"),
                        st.number_input(f"{param} max", 300, 500, 350, key=f"{param}_max")
                    )
                elif param == "P1":
                    param_ranges[param] = (
                        st.number_input(f"{param} min", 1e5, 4e5, 1e5, key=f"{param}_min"),
                        st.number_input(f"{param} max", 2e5, 5e5, 3e5, key=f"{param}_max")
                    )
                elif param == "Ratio de pression":
                    param_ranges[param] = (
                        st.slider(f"{param} min", 2, 15, 3, key=f"{param}_min"),
                        st.slider(f"{param} max", 5, 20, 15, key=f"{param}_max")
                    )
                else:  # T3
                    param_ranges[param] = (
                        st.number_input(f"{param} min", 800, 1300, 900, help="Min: 800K", key=f"{param}_min"),
                        st.number_input(f"{param} max", 900, 1500, 1300, key=f"{param}_max")
                    )
        
        param_points = st.slider("Nombre de points par paramètre", 5, 50, 15, key="param_points_brayton")

# ========================================
# LANCER L'ETUDE PARAMETRIQUE
# ========================================
run_study = st.button("Lancer l'étude paramétrique")

if run_study:
    if not params_to_vary:
        st.error("Veuillez sélectionner au moins un paramètre à faire varier")
    else:
        # ----------------------------------------
        # Initialisation de la simulation
        # ----------------------------------------
        st.info("Simulation en cours...")
        progress_bar = st.progress(0)
        timer_placeholder = st.empty()
        table_placeholder = st.empty()
        
        # Dictionnaire pour stocker les résultats de chaque paramètre
        all_results = {}
        start_time = time.time()
        
        if choix_cycle == "Cycle d'Ericsson":
            # ----------------------------------------
            # SIMULATION ERICSSON
            # Boucle sur chaque paramètre sélectionné
            # ----------------------------------------
            total_params = len(params_to_vary)
            total_sims = total_params * param_points
            sim_count = 0
            
            for param_to_vary in params_to_vary:
                param_min, param_max = param_ranges[param_to_vary]
                param_values = np.linspace(param_min, param_max, param_points)
                results_list = []
                
                for param_val in param_values:
                    # Assigner la valeur au paramètre variable
                    if param_to_vary == "T_min":
                        T_min_var, T_max_var, P_min_var, P_max_var = param_val, T_max_fixed, P_min_fixed, P_max_fixed
                    elif param_to_vary == "T_max":
                        T_min_var, T_max_var, P_min_var, P_max_var = T_min_fixed, param_val, P_min_fixed, P_max_fixed
                    elif param_to_vary == "P_min":
                        T_min_var, T_max_var, P_min_var, P_max_var = T_min_fixed, T_max_fixed, param_val, P_max_fixed
                    else:  # P_max
                        T_min_var, T_max_var, P_min_var, P_max_var = T_min_fixed, T_max_fixed, P_min_fixed, param_val
                    
                    if T_max_var <= T_min_var or P_max_var <= P_min_var:
                        continue
                    
                    try:
                        res = Ericsson.cycle_ericsson(T_min_var, T_max_var, P_min_var, P_max_var, show_plot=False)
                        ener = res["energetics"]
                        results_list.append({
                            param_to_vary: param_val,
                            "Qin (kJ/kg)": ener.get("Q_in", 0)/1000,
                            "Qout (kJ/kg)": ener.get("Q_out", 0)/1000,
                            "Wnet (kJ/kg)": ener.get("W_cycle", 0)/1000,
                            "Rendement (%)": ener.get("eta", 0)*100
                        })
                    except Exception as e:
                        continue
                    
                    sim_count += 1
                    
                    # Mise à jour en temps réel (tableau uniquement)
                    df = pd.DataFrame(results_list)
                    table_placeholder.dataframe(df)
                    
                    progress_bar.progress(int(100*sim_count/total_sims))
                    elapsed = time.time() - start_time
                    remaining = (elapsed/sim_count)*(total_sims-sim_count) if sim_count > 0 else 0
                    timer_placeholder.info(f"Temps restant estimé : {int(remaining)} s")
                
                all_results[param_to_vary] = pd.DataFrame(results_list)
            
            # Effacer le tableau intermédiaire
            table_placeholder.empty()
            
            # ========================================
            # AFFICHAGE DES VALEURS OPTIMALES
            # ========================================
            st.subheader("Valeurs optimales par paramètre")
            optimal_data = []
            for param_name, df in all_results.items():
                idx_best = df["Rendement (%)"].idxmax()
                best_row = df.loc[idx_best]
                optimal_data.append({
                    "Paramètre varié": param_name,
                    "Valeur optimale": best_row[param_name],
                    "Qin (kJ/kg)": round(best_row["Qin (kJ/kg)"], 2),
                    "Qout (kJ/kg)": round(best_row["Qout (kJ/kg)"], 2),
                    "Wnet (kJ/kg)": round(best_row["Wnet (kJ/kg)"], 2),
                    "Rendement (%)": round(best_row["Rendement (%)"], 2)
                })
            st.dataframe(pd.DataFrame(optimal_data), use_container_width=True)
            
            # ========================================
            # GRAPHIQUES SUPERPOSES (Ericsson)
            # ========================================
            st.subheader("Graphiques superposés")
            
            # Unités pour chaque paramètre
            unit_labels = {"T_min": "K", "T_max": "K", "P_min": "bar", "P_max": "bar"}
            
            # --- Graphiques Qin et Qout côte à côte ---
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Chaleur absorbée Qin (kJ/kg)**")
                chart_qin = create_overlay_chart(all_results, "Qin (kJ/kg)", "Qin", unit_labels, "Qin (kJ/kg)")
                st.altair_chart(chart_qin, use_container_width=True)
            with col2:
                st.markdown("**Chaleur rejetée Qout (kJ/kg)**")
                chart_qout = create_overlay_chart(all_results, "Qout (kJ/kg)", "Qout", unit_labels, "Qout (kJ/kg)")
                st.altair_chart(chart_qout, use_container_width=True)
            
            # --- Graphique Travail net ---
            st.markdown("**Travail net Wnet (kJ/kg)**")
            chart_wnet = create_overlay_chart(all_results, "Wnet (kJ/kg)", "Wnet", unit_labels, "Wnet (kJ/kg)")
            st.altair_chart(chart_wnet, use_container_width=True)
            
            # --- Graphique Rendement ---
            st.markdown("**Rendement thermique (%)**")
            chart_eta = create_overlay_chart(all_results, "Rendement (%)", "Rendement", unit_labels, "Rendement (%)")
            st.altair_chart(chart_eta, use_container_width=True)
            
            st.caption("Survolez les graphiques pour voir les valeurs exactes. Tous les paramètres s'affichent au même point.")
        
        elif choix_cycle == "Cycle de Stirling":
            # ----------------------------------------
            # SIMULATION STIRLING
            # Boucle sur chaque paramètre sélectionné
            # ----------------------------------------
            total_params = len(params_to_vary)
            total_sims = total_params * param_points
            sim_count = 0
            
            for param_to_vary in params_to_vary:
                param_min, param_max = param_ranges[param_to_vary]
                param_values = np.linspace(param_min, param_max, param_points)
                results_list = []
                
                for param_val in param_values:
                    # Assigner la valeur au paramètre variable
                    if param_to_vary == "T_min":
                        T_min_var, T_max_var, V_min_var, V_max_var = param_val, T_max_fixed_st, V_min_fixed_st, V_max_fixed_st
                    elif param_to_vary == "T_max":
                        T_min_var, T_max_var, V_min_var, V_max_var = T_min_fixed_st, param_val, V_min_fixed_st, V_max_fixed_st
                    elif param_to_vary == "V_min":
                        T_min_var, T_max_var, V_min_var, V_max_var = T_min_fixed_st, T_max_fixed_st, param_val, V_max_fixed_st
                    else:  # V_max
                        T_min_var, T_max_var, V_min_var, V_max_var = T_min_fixed_st, T_max_fixed_st, V_min_fixed_st, param_val
                    
                    # Validation des paramètres
                    if T_max_var <= T_min_var or V_max_var <= V_min_var:
                        continue
                    
                    try:
                        res = Stirling.cycle_stirling(T_min_var, T_max_var, V_min_var, V_max_var, show_plot=False)
                        ener = res["energetics"]
                        results_list.append({
                            param_to_vary: param_val,
                            "Qin (kJ/kg)": ener.get("Q_in", 0)/1000,
                            "Qout (kJ/kg)": ener.get("Q_out", 0)/1000,
                            "Wnet (kJ/kg)": ener.get("W_cycle", 0)/1000,
                            "Q_regen (kJ/kg)": ener.get("Q_regen", 0)/1000,
                            "Rendement (%)": ener.get("eta", 0)*100,
                            "Carnot (%)": ener.get("eta_carnot", 0)*100
                        })
                    except Exception as e:
                        continue
                    
                    sim_count += 1
                    
                    # Mise à jour en temps réel (tableau uniquement)
                    df = pd.DataFrame(results_list)
                    table_placeholder.dataframe(df)
                    
                    progress_bar.progress(int(100*sim_count/total_sims))
                    elapsed = time.time() - start_time
                    remaining = (elapsed/sim_count)*(total_sims-sim_count) if sim_count > 0 else 0
                    timer_placeholder.info(f"Temps restant estimé : {int(remaining)} s")
                
                all_results[param_to_vary] = pd.DataFrame(results_list)
            
            # Effacer le tableau intermédiaire
            table_placeholder.empty()
            
            # ========================================
            # AFFICHAGE DES VALEURS OPTIMALES (Stirling)
            # ========================================
            st.subheader("Valeurs optimales par paramètre")
            optimal_data = []
            for param_name, df in all_results.items():
                idx_best = df["Rendement (%)"].idxmax()
                best_row = df.loc[idx_best]
                optimal_data.append({
                    "Paramètre varié": param_name,
                    "Valeur optimale": best_row[param_name],
                    "Qin (kJ/kg)": round(best_row["Qin (kJ/kg)"], 2),
                    "Qout (kJ/kg)": round(best_row["Qout (kJ/kg)"], 2),
                    "Wnet (kJ/kg)": round(best_row["Wnet (kJ/kg)"], 2),
                    "Q_regen (kJ/kg)": round(best_row["Q_regen (kJ/kg)"], 2),
                    "Rendement (%)": round(best_row["Rendement (%)"], 2),
                    "Carnot (%)": round(best_row["Carnot (%)"], 2)
                })
            st.dataframe(pd.DataFrame(optimal_data), use_container_width=True)
            
            # ========================================
            # GRAPHIQUES SUPERPOSES (Stirling)
            # ========================================
            st.subheader("Graphiques superposés")
            
            # Unités pour chaque paramètre
            unit_labels = {"T_min": "K", "T_max": "K", "V_min": "m³/kg", "V_max": "m³/kg"}
            
            # --- Graphiques Qin et Qout côte à côte ---
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Chaleur absorbée Qin (kJ/kg)**")
                chart_qin = create_overlay_chart(all_results, "Qin (kJ/kg)", "Qin", unit_labels, "Qin (kJ/kg)")
                st.altair_chart(chart_qin, use_container_width=True)
            with col2:
                st.markdown("**Chaleur rejetée Qout (kJ/kg)**")
                chart_qout = create_overlay_chart(all_results, "Qout (kJ/kg)", "Qout", unit_labels, "Qout (kJ/kg)")
                st.altair_chart(chart_qout, use_container_width=True)
            
            # --- Graphique Travail net ---
            st.markdown("**Travail net Wnet (kJ/kg)**")
            chart_wnet = create_overlay_chart(all_results, "Wnet (kJ/kg)", "Wnet", unit_labels, "Wnet (kJ/kg)")
            st.altair_chart(chart_wnet, use_container_width=True)
            
            # --- Graphique Rendement ---
            st.markdown("**Rendement thermique (%)**")
            chart_eta = create_overlay_chart(all_results, "Rendement (%)", "Rendement", unit_labels, "Rendement (%)")
            st.altair_chart(chart_eta, use_container_width=True)
            
            st.caption("Survolez les graphiques pour voir les valeurs exactes. Tous les paramètres s'affichent au même point.")
        
        else:  # Brayton
            # ----------------------------------------
            # SIMULATION BRAYTON
            # Boucle sur chaque paramètre sélectionné
            # ----------------------------------------
            total_params = len(params_to_vary)
            total_sims = total_params * param_points
            sim_count = 0
            
            for param_to_vary in params_to_vary:
                param_min, param_max = param_ranges[param_to_vary]
                param_values = np.linspace(param_min, param_max, int(param_points))
                results_list = []
                
                for param_val in param_values:
                    # Assigner la valeur au paramètre variable
                    if param_to_vary == "T1":
                        T1_var, P1_var, PR_var, T3_var = param_val, P1_fixed, PR_fixed, T3_fixed
                    elif param_to_vary == "P1":
                        T1_var, P1_var, PR_var, T3_var = T1_fixed, param_val, PR_fixed, T3_fixed
                    elif param_to_vary == "Ratio de pression":
                        T1_var, P1_var, PR_var, T3_var = T1_fixed, P1_fixed, int(param_val), T3_fixed
                    else:  # T3
                        T1_var, P1_var, PR_var, T3_var = T1_fixed, P1_fixed, PR_fixed, param_val
                    
                    # Validation des paramètres
                    if T3_var <= T1_var:
                        continue
                    
                    try:
                        res = Brayton.cycle_brayton(T1_var, P1_var, PR_var, T3_var, show_plot=False)
                        ener = res["energetics"]
                        results_list.append({
                            param_to_vary: param_val,
                            "Wcomp (kJ/kg)": ener.get("W_comp", 0)/1000,
                            "Wturb (kJ/kg)": ener.get("W_turb", 0)/1000,
                            "Wnet (kJ/kg)": ener.get("W_net", 0)/1000,
                            "Qin (kJ/kg)": ener.get("Q_in", 0)/1000,
                            "Rendement (%)": ener.get("eta", 0)*100
                        })
                    except Exception as e:
                        continue
                    
                    sim_count += 1
                    
                    # Mise à jour en temps réel (tableau uniquement)
                    df = pd.DataFrame(results_list)
                    table_placeholder.dataframe(df)
                    
                    progress_bar.progress(int(100*sim_count/total_sims))
                    elapsed = time.time() - start_time
                    remaining = (elapsed/sim_count)*(total_sims-sim_count) if sim_count > 0 else 0
                    timer_placeholder.info(f"Temps restant estimé : {int(remaining)} s")
                
                all_results[param_to_vary] = pd.DataFrame(results_list)
            
            # Effacer le tableau intermédiaire
            table_placeholder.empty()
            
            # ========================================
            # AFFICHAGE DES VALEURS OPTIMALES
            # ========================================
            st.subheader("Valeurs optimales par paramètre")
            optimal_data = []
            for param_name, df in all_results.items():
                idx_best = df["Rendement (%)"].idxmax()
                best_row = df.loc[idx_best]
                optimal_data.append({
                    "Paramètre varié": param_name,
                    "Valeur optimale": best_row[param_name],
                    "Wcomp (kJ/kg)": round(best_row["Wcomp (kJ/kg)"], 2),
                    "Wturb (kJ/kg)": round(best_row["Wturb (kJ/kg)"], 2),
                    "Wnet (kJ/kg)": round(best_row["Wnet (kJ/kg)"], 2),
                    "Qin (kJ/kg)": round(best_row["Qin (kJ/kg)"], 2),
                    "Rendement (%)": round(best_row["Rendement (%)"], 2)
                })
            st.dataframe(pd.DataFrame(optimal_data), use_container_width=True)
            
            # ========================================
            # GRAPHIQUES SUPERPOSES (Brayton)
            # ========================================
            st.subheader("Graphiques superposés")
            
            # Unités pour chaque paramètre
            unit_labels = {"T1": "K", "T3": "K", "P1": "bar", "Ratio de pression": ""}
            
            # --- Graphiques Wcomp et Wturb côte à côte ---
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Travail compresseur Wcomp (kJ/kg)**")
                chart_wcomp = create_overlay_chart(all_results, "Wcomp (kJ/kg)", "Wcomp", unit_labels, "Wcomp (kJ/kg)")
                st.altair_chart(chart_wcomp, use_container_width=True)
            with col2:
                st.markdown("**Travail turbine Wturb (kJ/kg)**")
                chart_wturb = create_overlay_chart(all_results, "Wturb (kJ/kg)", "Wturb", unit_labels, "Wturb (kJ/kg)")
                st.altair_chart(chart_wturb, use_container_width=True)
            
            # --- Graphiques Wnet et Qin côte à côte ---
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Travail net Wnet (kJ/kg)**")
                chart_wnet = create_overlay_chart(all_results, "Wnet (kJ/kg)", "Wnet", unit_labels, "Wnet (kJ/kg)")
                st.altair_chart(chart_wnet, use_container_width=True)
            with col2:
                st.markdown("**Chaleur absorbée Qin (kJ/kg)**")
                chart_qin = create_overlay_chart(all_results, "Qin (kJ/kg)", "Qin", unit_labels, "Qin (kJ/kg)")
                st.altair_chart(chart_qin, use_container_width=True)
            
            # --- Graphique Rendement ---
            st.markdown("**Rendement thermique (%)**")
            chart_eta = create_overlay_chart(all_results, "Rendement (%)", "Rendement", unit_labels, "Rendement (%)")
            st.altair_chart(chart_eta, use_container_width=True)
            
            st.caption("Survolez les graphiques pour voir les valeurs exactes. Tous les paramètres s'affichent au même point.")
        
        # ========================================
        # EXPORT DES DONNEES EN CSV
        # ========================================
        st.subheader("Télécharger les données")
        combined_results = []
        for param_name, df in all_results.items():
            combined_results.append(df)
        
        if combined_results:
            full_df = pd.concat(combined_results, ignore_index=False)
            st.dataframe(full_df)
            
            # Bouton de téléchargement CSV
            csv_data = full_df.to_csv(index=False)
            st.download_button(
                label="Télécharger les résultats en CSV",
                data=csv_data,
                file_name="resultats_etude_parametrique.csv",
                mime="text/csv"
            )

# ================================
# Import CSV
# ================================

st.markdown("---")
st.header("Importer un CSV pour tracer les résultats existants")

csv_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])

if csv_file is not None:
    df_csv = pd.read_csv(csv_file)

    # Pagination pour gros fichiers
    st.subheader("Aperçu des données importées (pagination)")
    page_size = st.number_input("Nombre de lignes par page", min_value=5, max_value=100, value=20)
    total_pages = int(np.ceil(len(df_csv) / page_size))
    page_num = st.slider("Page", 1, total_pages, 1)
    start_idx = (page_num - 1) * page_size
    end_idx = start_idx + page_size
    st.dataframe(df_csv.iloc[start_idx:end_idx])

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




