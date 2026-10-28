import pandas as pd, os
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load & Train model (once)
@st.cache_resource
def load_model():
    csv_file = None
    for root, dirs, files in os.walk("."):
        for f in files:
            if f.startswith("train") and f.endswith(".csv"):
                csv_file = os.path.join(root, f)
    df = pd.read_csv(csv_file)
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    df = df.drop(columns=['Loan_ID'])
    df_encoded = pd.get_dummies(df, columns=['Gender','Married','Dependents','Education','Self_Employed','Property_Area'], drop_first=True)
    df_encoded['Loan_Status'] = df_encoded['Loan_Status'].map({'Y':1, 'N':0})
    X = df_encoded.drop(columns=['Loan_Status'])
    y = df_encoded['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_s, y_train)
    return model, scaler, X.columns.tolist()

model, scaler, feature_cols = load_model()

# UI
st.title("🏦 Loan Approval Prediction App")
st.write("Enter applicant information")

col1, col2 = st.columns(2)
with col1:
    Gender = st.selectbox("Gender", ["Male","Female"])
    Married = st.selectbox("Married", ["Yes","No"])
    Dependents = st.selectbox("Dependents", ["0","1","2","3+"])
    Education = st.selectbox("Education", ["Graduate","Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["No","Yes"])
with col2:
    ApplicantIncome = st.number_input("Applicant Income", 1000, 50000, 5000)
    CoapplicantIncome = st.number_input("Coapplicant Income", 0, 20000, 0)
    LoanAmount = st.number_input("Loan Amount", 10, 700, 150)
    Loan_Amount_Term = st.selectbox("Loan Term", [360, 180, 120, 84, 60])
    Credit_History = st.selectbox("Credit History", [1,0])
    Property_Area = st.selectbox("Property Area", ["Urban","Semiurban","Rural"])

if st.button("Predict Loan Approval"):
    # Build input dict matching training columns
    input_dict = {
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Gender_Male': 1 if Gender=="Male" else 0,
        'Married_Yes': 1 if Married=="Yes" else 0,
        'Dependents_1': 1 if Dependents=="1" else 0,
        'Dependents_2': 1 if Dependents=="2" else 0,
        'Dependents_3+': 1 if Dependents=="3+" else 0,
        'Education_Not Graduate': 1 if Education=="Not Graduate" else 0,
        'Self_Employed_Yes': 1 if Self_Employed=="Yes" else 0,
        'Property_Area_Semiurban': 1 if Property_Area=="Semiurban" else 0,
        'Property_Area_Urban': 1 if Property_Area=="Urban" else 0,
    }
    # Ensure order
    input_df = pd.DataFrame([input_dict])
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols]
    
    input_scaled = scaler.transform(input_df)
    prob = model.predict_proba(input_scaled)[0,1]
    pred = model.predict(input_scaled)[0]
    
    st.divider()
    st.metric("Approval Probability", f"{prob*100:.2f}%")
    if pred==1:
        st.success(f"✅ APPROVED - Probability {prob:.3f}")
    else:
        st.error(f"❌ REJECTED - Probability {prob:.3f}")
    
    st.write(f"Predicted Class: {pred} | Expected: {'Approved if Credit_History=1' if Credit_History==1 else 'Likely Reject'}")