import pandas as pd

csv_file = "archive/train_u6ljuX_CVtuZ9i.csv"

import os
for root, dirs, files in os.walk("."):
    for f in files:
        if f.startswith("train") and f.endswith(".csv"):
            csv_file = os.path.join(root, f)

print(f"Using file: {csv_file}\n")
df = pd.read_csv(csv_file)

print("===== 1. DATASET PREVIEW (first five rows) =====")
print(df.head())
print("\n")

print("===== 2. SHAPE (rows, columns) =====")
print(f"Shape: {df.shape}")
print("\n")

print("===== 3. COLUMNS =====")
for col in df.columns:
    print(f"- {col}")
print("\n")

print("===== 4. DATA TYPES =====")
print(df.dtypes)
print("\n")

print("===== 5. APPROVAL DISTRIBUTION =====")
if 'Loan_Status' in df.columns:
    print(df['Loan_Status'].value_counts())
    print(df['Loan_Status'].value_counts(normalize=True))
else:
    print(df['Loan_Status'].value_counts() if 'Loan_Status' in df.columns else df.iloc[:,-1].value_counts())

print("\n")



# ===== 6. Missing-value analysis =====
print("===== 6. MISSING-VALUE ANALYSIS =====")
print(df.isnull().sum())
print(f"\nTotal missing: {df.isnull().sum().sum()}")
print("\n")

# Duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")
print("\n")

# ===== 7. Cleaning results =====
print("===== 7. CLEANING RESULTS =====")

df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

# Remove duplicates if any
df = df.drop_duplicates()

print("After cleaning:")
print(df.isnull().sum())
print(f"Shape after cleaning: {df.shape}")
print("\n")

# ===== 8. Encoding =====
print("===== 8. ENCODING (Categorical -> Numeric for Logistic Regression) =====")
# Drop Loan_ID
df_clean = df.drop(columns=['Loan_ID'])

# Show before encoding
print("Before encoding columns:", df_clean.columns.tolist())

# One-hot encode + Label encode for Logistic Regression
df_encoded = pd.get_dummies(df_clean, columns=['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status'], drop_first=True)

print("\nAfter encoding:")
print(df_encoded.head())
print(f"\nEncoded shape: {df_encoded.shape}")
print(f"Encoded columns: {df_encoded.columns.tolist()}")
print("\n")

# ===== 9. Final dataset =====
print("===== 9. FINAL DATASET READY FOR LOGISTIC REGRESSION =====")
print(df_encoded.head())
print("\nData types:")
print(df_encoded.dtypes)
print(f"\nFinal dataset shape: {df_encoded.shape}")
print("No missing values:", df_encoded.isnull().sum().sum() == 0)

df_encoded.to_csv("final_loan_dataset_for_model.csv", index=False)
print("\nSaved as final_loan_dataset_for_model.csv")