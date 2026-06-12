import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Loan Default Predictor", page_icon="🏦")
st.title("🏦 Nigerian Loan Default Predictor")
st.write("Enter applicant details below to predict loan default risk.")

# Load and prepare data
@st.cache_data
def train_model():
    df = pd.read_csv("nigerian loan default dataset.csv")
    df = df.drop(columns=['applicant_name'])
    cat_cols = ['gender','state','employment_type','loan_purpose','collateral_type']
    le = LabelEncoder()
    encoders = {}
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
        encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    X = df.drop(columns=['loan_default'])
    y = df['loan_default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, encoders

model, encoders = train_model()

# Input form
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 22, 65, 35)
    state = st.selectbox("State", ["Lagos","Abuja","Kano","Rivers","Oyo","Kaduna",
                                    "Anambra","Enugu","Delta","Borno"])
    employment_type = st.selectbox("Employment Type", ["Salary Earner","Self-Employed",
                                    "Civil Servant","Farmer","Trader","Contractor"])
    loan_purpose = st.selectbox("Loan Purpose", ["Business Capital","Agricultural Input",
                                    "School Fees","Home Improvement","Medical",
                                    "Vehicle Purchase","Asset Acquisition"])

with col2:
    monthly_income = st.number_input("Monthly Income (₦)", 30000, 500000, 150000, step=10000)
    loan_amount = st.number_input("Loan Amount (₦)", 50000, 5000000, 500000, step=50000)
    loan_tenure = st.selectbox("Loan Tenure (Months)", [6,12,18,24,36,48,60])
    credit_score = st.slider("Credit Score", 300, 850, 600)
    previous_default = st.selectbox("Previous Default?", ["No", "Yes"])
    collateral = st.selectbox("Collateral Type", ["Land","Vehicle","Guarantor",
                                                   "Fixed Deposit","None"])

# Encode inputs
def encode(col, val):
    return encoders[col].get(val, 0)

input_data = pd.DataFrame([{
    'gender': encode('gender', gender),
    'age': age,
    'state': encode('state', state),
    'employment_type': encode('employment_type', employment_type),
    'monthly_income_naira': monthly_income,
    'loan_amount_naira': loan_amount,
    'loan_purpose': encode('loan_purpose', loan_purpose),
    'loan_tenure_months': loan_tenure,
    'credit_score': credit_score,
    'previous_default': 1 if previous_default == "Yes" else 0,
    'collateral_type': encode('collateral_type', collateral)
}])

# Predict
if st.button("🔍 Predict Default Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ HIGH RISK — This applicant is likely to DEFAULT")
    else:
        st.success(f"✅ LOW RISK — This applicant is likely to REPAY")

    st.metric("Default Probability", f"{probability*100:.1f}%")