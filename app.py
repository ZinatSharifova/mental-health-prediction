import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Mental Health Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# SABİT DƏYƏRLƏR (TƏLİMDƏN GƏLƏN)
# ==============================
# LightGBM üçün təlimdə tapılmış optimal hədd (Recall 0.63 üçün)
# Sizin LightGBM nəticələrinizdə: 0.4479
OPTIMAL_THRESHOLD = 0.4479 

# ==============================
# LOAD MODEL & ARTIFACTS
# ==============================
@st.cache_resource
def load_models():
    # Model LightGBM olmalıdır
    model = joblib.load("lgbm_best_model.pkl") 
    scaler = joblib.load("final_scaler.pkl")
    te_maps = joblib.load("final_target_encoding_maps.pkl")
    return model, scaler, te_maps

# Model, Scaler və Target Encoding xəritələrini yükləyirik
try:
    model, scaler, te_maps = load_models()
except FileNotFoundError:
    st.error("Model və ya preprocessinq faylları tapılmadı (lgbm_best_model.pkl, final_scaler.pkl, final_target_encoding_maps.pkl). Zəhmət olmasa, əvvəlki addımları işlədin.")
    st.stop()


# ==============================
# FEATURE DEFINITIONS
# ==============================
numeric_features = [
    'net_personal_income', 'age',
    'moderate_days_per_week', 'vigorous_days_per_week',
    'household_size', 'number_of_children'
]

binary_features = ['smoking_status', 'drinks_alcohol_past12m', 'gender']

categorical_features = [
    'lack_companionship','loneliness_frequency','employment_status_binned', 
    'current_financial_situation_binned','future_financial_outlook_binned', 
    'marital_status_binned','education_level_binned', 'alcohol_frequency_12m', 
    'drinks_per_typical_day_binned', 'urban_rural', 'country',
    'income_satisfaction_binned', 'leisure_satisfaction_binned', 'job_satisfaction_binned'
]

# EXACT category names from data (post-binning)
CATEGORY_OPTIONS = {
    'lack_companionship': ['Hardly ever/never', 'Some of the time', 'Often'],
    'loneliness_frequency': ['Hardly ever/never', 'Some of the time', 'Often'],
    'employment_status_binned': ['Employed', 'Retired/Elderly', 'Student/Apprentice', 
                                 'Home/Family', 'Unemployed/Other', 'Sick/Disabled'],
    'current_financial_situation_binned': ['Comfortable', 'Moderate', 'Struggling'],
    'future_financial_outlook_binned': ['About the same', 'Better off', 'Negative/Uncertain'],
    'marital_status_binned': ['Married/Civil partner', 'Never married', 'Living as couple', 
                               'Previously Married', 'Widowed'],
    'education_level_binned': ['Secondary Education', 'Higher Education', 'None of the above', 
                                 'Inapplicable', 'Vocational/Medical'],
    'alcohol_frequency_12m': ['Non-Drinker', '2-4 times a month', 'Monthly or less', 
                                 '2-3 times a week', '4+ times a week'],
    'drinks_per_typical_day_binned': ['Light Drinking', 'Non Drinker', 'Moderate Drinking', 'Heavy Drinking'],
    'urban_rural': ['Urban', 'Rural'],
    'country': ['England', 'Scotland', 'Wales', 'Northern Ireland'],
    'income_satisfaction_binned': ['Satisfied', 'Neutral', 'Dissatisfied'],
    'leisure_satisfaction_binned': ['Satisfied', 'Neutral', 'Dissatisfied'],
    'job_satisfaction_binned': ['High', 'Medium', 'Low', 'Unemployed']
}

# Fruit mapping (0,1,2,3 as in training code)
fruit_mapping = {
    'Never': 0,
    '1-3 Days': 1,
    '4-6 Days': 2,
    'Every Day': 3,
}

# ==============================
# MAIN UI
# ==============================
st.title("Mental Health Risk Prediction")
st.markdown("Predict mental health risk using LightGBM model trained on Understanding Society data.")
st.info(f"Fill out the form in the sidebar and click Predict. **Prediction uses Optimal Threshold: {OPTIMAL_THRESHOLD:.4f}**")

# ==============================
# SIDEBAR - USER INPUT FORM
# ==============================
st.sidebar.header("Personal Information")

input_data = {}

# --- DEMOGRAPHICS ---
st.sidebar.subheader("Demographics")

input_data['age'] = st.sidebar.number_input(
    "Age", 
    min_value=16, 
    max_value=100, 
    value=30, 
    step=1
)

input_data['gender'] = st.sidebar.selectbox("Gender", options=["Male", "Female"])

input_data['marital_status_binned'] = st.sidebar.selectbox(
    "Marital Status",
    options=CATEGORY_OPTIONS['marital_status_binned']
)

input_data['household_size'] = st.sidebar.slider("Household Size", 1, 20, 2, 1)
input_data['number_of_children'] = st.sidebar.slider("Number of Children", 0, 10, 0, 1)

input_data['education_level_binned'] = st.sidebar.selectbox(
    "Education Level",
    options=CATEGORY_OPTIONS['education_level_binned']
)

input_data['urban_rural'] = st.sidebar.selectbox("Location", options=CATEGORY_OPTIONS['urban_rural'])
input_data['country'] = st.sidebar.selectbox("Country", options=CATEGORY_OPTIONS['country'])

# --- ECONOMIC ---
st.sidebar.subheader("Economic Situation")

input_data['net_personal_income'] = st.sidebar.number_input(
    "Monthly Income (£)",
    min_value=0,
    max_value=20000,
    value=1000,
    step=100
)

input_data['employment_status_binned'] = st.sidebar.selectbox(
    "Employment Status",
    options=CATEGORY_OPTIONS['employment_status_binned']
)

input_data['current_financial_situation_binned'] = st.sidebar.selectbox(
    "Current Financial Situation",
    options=CATEGORY_OPTIONS['current_financial_situation_binned']
)

input_data['future_financial_outlook_binned'] = st.sidebar.selectbox(
    "Future Financial Outlook",
    options=CATEGORY_OPTIONS['future_financial_outlook_binned']
)

input_data['income_satisfaction_binned'] = st.sidebar.selectbox(
    "Income Satisfaction",
    options=CATEGORY_OPTIONS['income_satisfaction_binned']
)

input_data['job_satisfaction_binned'] = st.sidebar.selectbox(
    "Job Satisfaction",
    options=CATEGORY_OPTIONS['job_satisfaction_binned']
)

# --- LIFESTYLE ---
st.sidebar.subheader("Lifestyle & Health")

input_data['smoking_status'] = st.sidebar.selectbox("Do you smoke?", options=["No", "Yes"])

input_data['drinks_alcohol_past12m'] = st.sidebar.selectbox(
    "Consumed alcohol in past 12 months?",
    options=["No", "Yes"]
)

input_data['alcohol_frequency_12m'] = st.sidebar.selectbox(
    "Alcohol Frequency",
    options=CATEGORY_OPTIONS['alcohol_frequency_12m']
)

input_data['drinks_per_typical_day_binned'] = st.sidebar.selectbox(
    "Drinks per Typical Day",
    options=CATEGORY_OPTIONS['drinks_per_typical_day_binned']
)

fruit_choice = st.sidebar.selectbox("Fruit consumption per week", options=list(fruit_mapping.keys()))
input_data['fruit_days_per_week'] = fruit_mapping[fruit_choice]

input_data['moderate_days_per_week'] = st.sidebar.slider("Moderate Exercise (days/week)", 0, 7, 0, 1)
input_data['vigorous_days_per_week'] = st.sidebar.slider("Vigorous Exercise (days/week)", 0, 7, 0, 1)

# --- SOCIAL WELLBEING ---
st.sidebar.subheader("Social Wellbeing")

input_data['loneliness_frequency'] = st.sidebar.selectbox(
    "How often do you feel lonely?",
    options=CATEGORY_OPTIONS['loneliness_frequency']
)

input_data['lack_companionship'] = st.sidebar.selectbox(
    "How often do you lack companionship?",
    options=CATEGORY_OPTIONS['lack_companionship']
)

input_data['leisure_satisfaction_binned'] = st.sidebar.selectbox(
    "Leisure Satisfaction",
    options=CATEGORY_OPTIONS['leisure_satisfaction_binned']
)

# ==============================
# PREPROCESSING (EXACT SAME AS TRAINING)
# ==============================
def preprocess_input(data):
    """Exact same preprocessing as training code"""
    df = pd.DataFrame([data])
    
    # Binary encoding (EXACT same as training)
    df['gender'] = (df['gender'] == 'Male').astype(int)
    df['smoking_status'] = (df['smoking_status'].astype(str).str.lower() == 'yes').astype(int)
    df['drinks_alcohol_past12m'] = (df['drinks_alcohol_past12m'].astype(str).str.lower() == 'yes').astype(int)
    
    # Target encoding (use saved maps)
    for col in categorical_features:
        if col in te_maps:
            mapping = te_maps[col]
            global_mean = np.mean(list(mapping.values()))
            df[col] = df[col].map(mapping).fillna(global_mean)
        else:
            df[col] = 0.0
    
    # Scaling (use saved scaler)
    df[numeric_features] = scaler.transform(df[numeric_features])
    
    # Correct feature order
    all_features = numeric_features + binary_features + categorical_features + ['fruit_days_per_week']
    df = df[all_features]
    
    return df

# ==============================
# PREDICTION
# ==============================
if st.sidebar.button("Predict Mental Health Risk", type="primary"):
    
    try:
        processed_data = preprocess_input(input_data)
        
        # Prediction
        proba = model.predict_proba(processed_data)[0, 1]
        
        # ************************************************************
        # DƏYİŞİKLİK: Proqnozu optimal həddə görə təyin edirik
        prediction = (proba >= OPTIMAL_THRESHOLD).astype(int) 
        # ************************************************************
        
        # Display results
        st.markdown("---")
        st.header("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Probability", f"{proba*100:.1f}%")
        
        with col2:
            if prediction == 1:
                st.error("High Risk (Poor Mental Health)")
                risk_label = "Poor Mental Health"
            else:
                st.success("Low Risk (Good Mental Health)")
                risk_label = "Good Mental Health"
        
        with col3:
            if proba < OPTIMAL_THRESHOLD * 0.7:
                risk_category = "Low Risk (Well Below Threshold)"
                color = "green"
            elif proba < OPTIMAL_THRESHOLD * 1.2:
                risk_category = "Medium Risk (Near Threshold)"
                color = "orange"
            else:
                risk_category = "High Risk (Well Above Threshold)"
                color = "red"
            st.markdown(f"### :{color}[{risk_category}]")
        
        # Interpretation
        st.markdown("---")
        st.subheader("Interpretation")
        
        if prediction == 1:
            st.warning(f"""
            **Model Prediction: {risk_label}**
            
            Based on your information, the model indicates higher risk for poor mental health.
            
            - Risk probability: **{proba*100:.1f}%** (This is $\ge$ Optimal Threshold of {OPTIMAL_THRESHOLD})
            - Consider speaking with a healthcare professional
            - This is a screening tool, NOT a clinical diagnosis
            """)
        else:
            st.success(f"""
            **Model Prediction: {risk_label}**
            
            Based on your information, the model indicates lower risk for poor mental health.
            
            - Risk probability: **{proba*100:.1f}%** (This is < Optimal Threshold of {OPTIMAL_THRESHOLD})
            - Continue maintaining healthy lifestyle factors
            - Stay aware of your wellbeing
            """)
        
        # Risk factors
        if prediction == 1: # Model prediction is based on the optimal threshold
            st.markdown("---")
            st.subheader("Key Risk Factors")
            st.markdown("""
            Factors that may contribute to mental health risk:
            - Social isolation
            - Financial stress
            - Job dissatisfaction
            - Limited physical activity
            """)
        
        # Disclaimer
        st.markdown("---")
        st.info("""
        **Disclaimer:**
        
        This tool is for informational purposes only. It should not replace professional medical advice.
        
        Model Performance (LightGBM): Test AUC 79.25%, Test Recall 63.0%, Test F1-Score 64.8%
        """)
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.info("Please ensure all fields are filled correctly.")

else:
    st.markdown("---")
    st.info(f"""
    ### Get Started
    
    Fill out the form in the sidebar and click **Predict Mental Health Risk**.
    
    **About the Model (LightGBM):**
    - LightGBM algorithm
    - Trained on 32,000+ individuals
    - Test AUC: 79.25%
    - Prediction is based on the optimal threshold of **{OPTIMAL_THRESHOLD:.4f}**
    """)
    
    # Metrika dəyərləri LightGBM nəticələrinə uyğun dəyişdirilib
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC", "79.25%")
    col2.metric("Recall", "63.0%")
    col3.metric("F1", "64.8%")
    col4.metric("Accuracy", "74.0%")
