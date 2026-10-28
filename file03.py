import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Find train file
csv_file = None
for root, dirs, files in os.walk("."):
    for f in files:
        if f.startswith("train") and f.endswith(".csv"):
            csv_file = os.path.join(root, f)

df = pd.read_csv(csv_file)

# --- Cleaning (fixed version) ---
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

# --- Encoding ---
df = df.drop(columns=['Loan_ID'])
df_encoded = pd.get_dummies(df, columns=['Gender','Married','Dependents','Education','Self_Employed','Property_Area'], drop_first=True)
# Encode target Loan_Status Y/N -> 1/0
df_encoded['Loan_Status'] = df_encoded['Loan_Status'].map({'Y':1, 'N':0})

# Features and Target
X = df_encoded.drop(columns=['Loan_Status'])
y = df_encoded['Loan_Status']

print("===== 1. TRAIN/TEST SPLIT =====")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Total data: {X.shape[0]} rows")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train distribution: {y_train.value_counts().to_dict()}")
print(f"y_test distribution: {y_test.value_counts().to_dict()}")
print("\n")

print("===== 2. MODEL TRAINING =====")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Logistic Regression Model Trained Successfully!")
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")
print("\n")

print("===== 3. PREDICTIONS =====")
y_pred = model.predict(X_test)
print(f"First 10 Predictions: {y_pred[:10]}")
print(f"First 10 Actual: {y_test.values[:10]}")
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print("\n")

print("===== 4. PREDICTION PROBABILITIES =====")
y_proba = model.predict_proba(X_test)
print("Probabilities format: [Prob_No, Prob_Yes]")
print(f"First 10 Probabilities:\n{y_proba[:10]}")
# Show decision threshold example
df_result = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred,
    'Prob_No': y_proba[:,0].round(3),
    'Prob_Yes': y_proba[:,1].round(3)
})
print("\nDetailed Results (first 15 rows):")
print(df_result.head(15))