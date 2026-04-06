import json
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as light

st.markdown("""
<style>
/* Background */
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* Main container */
.main {
    background-color: rgba(0, 0, 0, 0.6);
    padding: 20px;
    border-radius: 15px;
}

/* Title */
h1 {
    color: #00FFD1;
    text-align: center;
    font-weight: bold;
}

/* Subheading */
h2, h3 {
    color: #00BFFF;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #00FFD1, #00BFFF);
    color: black;
    border-radius: 10px;
    font-weight: bold;
    padding: 10px;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
}

/* Metric cards */
[data-testid="metric-container"] {
    background-color: #111;
    border: 1px solid #00FFD1;
    padding: 10px;
    border-radius: 10px;
}

/* Alerts */
.stAlert {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

model = light.Booster(model_file="lightgbm_model_triage.txt")



with open("features.json", "r") as f:
    all_features = json.load(f)

st.set_page_config(page_title="AI Triage System", layout="wide")

st.title("AI Assisted Emergency Triage Prediction System")
st.markdown("Predict triage acuity using key clinical features")

st.sidebar.header("Patient Inputs")

# IMPORTANT FEATURES INPUT
arrival_mode = st.sidebar.selectbox("Arrival Mode (walk-in = 1 , police = 2 , ambulance = 3 , helicopter = 4 , brought by family = 5)", [1,2,3,4,5])

mental_status = st.sidebar.selectbox("Mental Status", [0,1,2])
ed_visits = st.sidebar.number_input("Prior ED Visits (12m)", 0, 50, 0)
admissions = st.sidebar.number_input("Prior Admissions (12m)", 0, 50, 0)
medications = st.sidebar.number_input("Active Medications", 0, 20, 0)
comorbidities = st.sidebar.number_input("Comorbidities (Other diseases except the primary disease)", 0, 10, 0)

systolic_bp = st.sidebar.number_input("Systolic BP", 50, 250, 120)
diastolic_bp = st.sidebar.number_input("Diastolic BP", 30, 150, 80)
resp_rate = st.sidebar.number_input("Respiratory Rate", 5, 40, 16)

pain_score = st.sidebar.slider("Pain Score", 0, 10, 3)

weight = st.sidebar.number_input("Weight (kg)", 20.0, 200.0, 70.0)
height = st.sidebar.number_input("Height (cm)", 100.0, 220.0, 170.0)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 0, 100, 25)
    spo2 = st.slider("SpO2 (%)", 50, 100, 98)

with col2:
    temp = st.slider("Temperature (°C)", 34.0, 42.0, 36.5)
    heart_rate = st.slider("Heart Rate (bpm)", 40, 180, 75)

shock_index = heart_rate / systolic_bp if systolic_bp != 0 else 0


if st.button("Predict"):

    
    data = {col: 0 for col in all_features}

    
    data.update({
        'arrival_mode': arrival_mode,
        'age': age,
        'mental_status_triage': mental_status,
        'num_prior_ed_visits_12m': ed_visits,
        'num_prior_admissions_12m': admissions,
        'num_active_medications': medications,
        'num_comorbidities': comorbidities,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'heart_rate': heart_rate,
        'respiratory_rate': resp_rate,
        'temperature_c': temp,
        'spo2': spo2,
        'pain_score': pain_score,
        'weight_kg': weight,
        'height_cm': height,
        'shock_index': shock_index
    })

    df = pd.DataFrame([data])
    df['hr_spo2_ratio'] = df['heart_rate'] / (df['spo2'] + 1)
    df['temp_hr'] = df['temperature_c'] * df['heart_rate']
    df['resp_spo2'] = df['respiratory_rate'] / (df['spo2'] + 1)
    
    df['pain_score'] = df['pain_score'] * 0.5
    
    df['spo2_critical'] = (df['spo2'] < 90).astype(int)
    
    df['temp_critical'] = (df['temperature_c'] > 39).astype(int)
    df['hr_critical'] = (df['heart_rate'] > 120).astype(int)
    
    df['severity_score'] = (
    df['spo2_critical'] + df['temp_critical'] + df['hr_critical']
)
    missing_cols = [col for col in all_features if col not in df.columns]
    for col in missing_cols:
        df[col] = 0
    
    extra_cols = [col for col in df.columns if col not in all_features]
    
    df = df.drop(columns=extra_cols)

    df = df[all_features]
    
    preds = model.predict(df)
    preds = np.argmax(preds, axis=1)[0]
    preds = preds + 1

    st.subheader("Patient Risk Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("SpO2", spo2)
    col2.metric("Heart Rate", heart_rate)
    col3.metric("Temperature", temp)

   
    st.subheader("Prediction Result")

    if preds == 1:
        st.error(f"Critical (Level 1)")
        st.progress(100)
    elif preds == 2:
        st.error(f"Very Urgent (Level 2)")
        st.progress(80)
    elif preds == 3:
        st.warning(f"Urgent (Level 3)")
        st.progress(60)
    elif preds == 4:
        st.info(f"Less Urgent (Level 4)")
        st.progress(40)
    else:
        st.success(f"Non-Urgent (Level 5)")
        st.progress(20)


st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    with st.expander("Learn more about Triage System"):

        st.markdown("## What is Triage?")
        st.markdown("""
        Triage is the process of prioritizing patients in an emergency department based on the severity of their condition. 
        It ensures that critically ill patients receive immediate care, while less urgent cases are attended appropriately.

        In high-pressure environments like emergency rooms, accurate triage is essential to save lives and optimize resource allocation.
        """)

        st.markdown("##  Problem Statement")
        st.markdown("""
        Emergency departments often face overcrowding, limited resources, and high patient inflow. 
        Manual triage systems depend heavily on human judgment, which can lead to:

        - Delays in identifying critical patients  
        - Inconsistent decision-making  
        - Increased risk of medical errors  

        This project uses LightGBM machine learning algorithm to predict triage acuity levels 
        based on patient clinical features, enabling faster and more reliable decision support.
        """)

with col2:
    if "preds" in locals():  # only show after prediction
        with st.expander("Patient Summary"):

            st.write({
                "Age": age,
                "Mental Status": mental_status,
                "Heart Rate": heart_rate,
                "SpO2": spo2,
                "Systolic BP": systolic_bp,
                "Diastolic BP": diastolic_bp,
                "Temperature": temp,
                "Pain Score": pain_score
            })


st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 10px;
    right: 20px;
    color: gray;
    font-size: 14px;
    opacity: 0.7;
}
</style>

<div class="footer">
    AI Triage System • Built with LightGBM
</div>
""", unsafe_allow_html=True)


