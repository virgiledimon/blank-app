import streamlit as st

st.title("FogCID-XAI")
st.write(
    "Explainable AI approach for impersonation attack detection in fog computing "
)


with st.form("input_params_form", clear_on_submit=False, enter_to_submit=True, border=True):
    st.write("Faites un choix de simulation")

    algo = st.selectbox("Agent de détection", ["Double SARSA"])
    xai_methode = st.selectbox("Méthode XAI", ["SHAP", "LIME", "Permutation Feature Importance"])
    episode_nbr = st.number_input("Nombre d'épisode", min_value=30)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Simuler")
    if submitted:
        st.write("episode_nbr", episode_nbr)

st.write("Dernières séquences de détection")

