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

st.title("Emergency Triage Prediction System")
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
    elif preds == 2:
        st.error(f"Very Urgent (Level 2)")
    elif preds == 3:
        st.warning(f"Urgent (Level 3)")
    elif preds == 4:
        st.info(f"Less Urgent (Level 4)")
    else:
        st.success(f"Non-Urgent (Level 5)")


st.markdown("---")
st.markdown("AI Triage System • Built with LightGBM")

