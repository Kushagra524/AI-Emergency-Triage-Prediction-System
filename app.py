import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as light


model = light.Booster(model_file="lightgbm_model_triage.txt")


all_features = [
    'site_id','triage_nurse_id','arrival_mode','age','age_group',
    'mental_status_triage','num_prior_ed_visits_12m',
    'num_prior_admissions_12m','num_active_medications',
    'num_comorbidities','systolic_bp','diastolic_bp',
    'mean_arterial_pressure','pulse_pressure','heart_rate',
    'respiratory_rate','temperature_c','spo2','gcs_total',
    'pain_score','weight_kg','height_cm','bmi','shock_index',
    'news2_score','day_sin','day_cos',
    'pain_location_back','pain_location_chest','pain_location_extremity',
    'pain_location_head','pain_location_multiple','pain_location_other',
    'pain_location_pelvis',
    'chief_complaint_system_critical',
    'chief_complaint_system_genitourinary',
    'chief_complaint_system_mild',
    'chief_complaint_system_moderate',
    'chief_complaint_system_special',
    'systolic_bp_missing','diastolic_bp_missing',
    'mean_arterial_pressure_missing','pulse_pressure_missing',
    'respiratory_rate_missing','temperature_c_missing',
    'shock_index_missing'
]


st.set_page_config(page_title="AI Triage System", layout="wide")

st.title("AI Assisted Emergency Triage Prediction System")

st.markdown("Predict triage acuity using key clinical features")

st.sidebar.header("Patient Inputs")

# IMPORTANT FEATURES INPUT
arrival_mode = st.sidebar.selectbox("Arrival Mode (walk-in = 1 , police = 2 , ambulance = 3 , helicopter = 4 , brought by family = 5)", [1,2,3,4,5])
age = st.sidebar.slider("Age", 0, 100, 30)
mental_status = st.sidebar.selectbox("Mental Status", [0,1,2])
ed_visits = st.sidebar.number_input("Prior ED Visits (12m)", 0, 50, 0)
admissions = st.sidebar.number_input("Prior Admissions (12m)", 0, 50, 0)
medications = st.sidebar.number_input("Active Medications", 0, 20, 0)
comorbidities = st.sidebar.number_input("Comorbidities (Other diseases except the primary disease)", 0, 10, 0)

systolic_bp = st.sidebar.number_input("Systolic BP", 50, 250, 120)
diastolic_bp = st.sidebar.number_input("Diastolic BP", 30, 150, 80)
heart_rate = st.sidebar.number_input("Heart Rate", 30, 200, 80)
resp_rate = st.sidebar.number_input("Respiratory Rate", 5, 40, 16)
temp = st.sidebar.number_input("Temperature (C)", 30.0, 42.0, 37.0)
spo2 = st.sidebar.number_input("SpO2", 50, 100, 98)

pain_score = st.sidebar.slider("Pain Score", 0, 10, 3)

weight = st.sidebar.number_input("Weight (kg)", 20.0, 200.0, 70.0)
height = st.sidebar.number_input("Height (cm)", 100.0, 220.0, 170.0)

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

    
    preds = model.predict(df)
    preds = np.argmax(preds, axis=1)[0]

   
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
    with st.expander("📖 Learn more about Triage System"):

        st.markdown("## 🏥 What is Triage?")
        st.markdown("""
        Triage is the process of prioritizing patients in an emergency department based on the severity of their condition. 
        It ensures that critically ill patients receive immediate care, while less urgent cases are attended appropriately.

        In high-pressure environments like emergency rooms, accurate triage is essential to save lives and optimize resource allocation.
        """)

        st.markdown("## ⚠️ Problem Statement")
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
        with st.expander("📋 Patient Summary"):

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

