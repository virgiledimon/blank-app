import streamlit as st

st.title("FogCID-XAI")
st.write(
    "Explainable AI approach for impersonation attack detection in fog computing "
)

st.form("input_params_form", clear_on_submit=False, enter_to_submit=True, border=True)

with st.form("my_form"):
    st.write("Inside the form")
    slider_val = st.slider("Form slider")
    checkbox_val = st.checkbox("Form checkbox")

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("slider", slider_val, "checkbox", checkbox_val)
st.write("Outside the form")