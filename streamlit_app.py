import streamlit as st
import pandas as pd
import time
from detection_agents.DoubleSarsaAgent import DoubleSarsaAgent  # à ajuster selon l'emplacement des agents
from explicitility_agents.SHAPAgent import SHAPAgent
from explicitility_agents.LIMEAgent import LIMEAgent
from explicitility_agents.PFIAgent import PFIAgent
from interpretability_agents.InterpretabilityAgent import InterpretabilityAgent

st.title("FogCID-XAI")
st.write("Explainable AI approach for impersonation attack detection in fog computing")

# Formulaire de choix de simulation
with st.form("input_params_form", clear_on_submit=False):
    st.write("Faites un choix de simulation")
    algo = st.selectbox("Agent de détection", ["Double SARSA"])
    xai_methode = st.selectbox("Méthode XAI", ["SHAP", "LIME", "Permutation Feature Importance"])
    episode_nbr = st.number_input("Nombre d'épisodes", min_value=30)
    submitted = st.form_submit_button("Simuler")

# Initialisation des agents
detection_agent = DoubleSarsaAgent() if algo == "Double SARSA" else None
xai_agent = {
    "SHAP": SHAPAgent(),
    "LIME": LIMEAgent(),
    "Permutation Feature Importance": PFIAgent()
}.get(xai_methode, None)
interpretability_agent = InterpretabilityAgent()

# Lorsque le formulaire est soumis
if submitted:
    # Exécution de l'agent de détection et collecte des données
    detection_agent.run_simulation(episode_nbr)  # Lancement de la simulation
    st.write(f"Lancement de la simulation pour {episode_nbr} épisodes")

    # Section dynamique pour afficher les dernières séquences de détection
    st.write("1- Tableau des dernières séquences de détection")
    detection_data_placeholder = st.empty()

    # Actualisation en continu des dernières détections
    for _ in range(episode_nbr):
        # On suppose que `fetch_latest_detections` est une fonction qui récupère les 10 dernières lignes de détection
        latest_detections = detection_agent.fetch_latest_detections(10)
        detection_df = pd.DataFrame(latest_detections, columns=["Timestamp", "Decision", "L_value", "Coord_R", "Coord_T"])
        detection_data_placeholder.write(detection_df)
        time.sleep(1)  # Pause pour actualisation en continu

    # Affichage de l'arbre de décision pour l'interprétabilité
    st.write("2- Représentation du modèle interprétable")
    interpretability_tree = interpretability_agent.get_decision_tree()
    st.graphviz_chart(interpretability_tree)  # Affiche l'arbre de décision

    # Affichage des résultats d'explicabilité en fonction de la méthode choisie
    st.write("3- Résultats d'explicabilité")
    explanation_results = xai_agent.generate_explanation(detection_agent)
    st.write(explanation_results)  # Affiche les résultats d'explicabilité

    # Pour SHAP : Affiche les valeurs de SHAP sous forme de graphique
    if xai_methode == "SHAP":
        shap_plot = xai_agent.plot_shap_values()
        st.pyplot(shap_plot)
