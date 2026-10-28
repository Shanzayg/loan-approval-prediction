# Loan Approval Prediction - End-to-End ML Project

### Problem Statement
Banks need to automate loan approval to reduce risk. Build a model that predicts if a loan will be approved based on applicant info.

### Dataset
- Source: Loan Prediction Dataset (614 rows, 13 columns)
- Train: 614 records from archive/train.csv
- Features: Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area
- Target: Loan_Status (Y/N)
- Missing: Gender, Married, Dependents, Self_Employed, LoanAmount, Loan_Amount_Term, Credit_History

### Preprocessing
1. Missing-value handling:
      - Categorical: Mode imputation (Gender, Married, Dependents, Self_Employed, Loan_Amount_Term, Credit_History)
      - Numerical: Median imputation (LoanAmount)
2. Duplicates: Removed 0 duplicates
3. Encoding:
      - Dropped Loan_ID
      - One-Hot Encoding for categorical columns with drop_first=True
      - Label Encoding for target: Y=1, N=0
4. Scaling: StandardScaler for Logistic Regression

### Model
- Model: Logistic Regression (max_iter=1000)
- Split: 80% Train (491), 20% Test (123), stratify=y, random_state=42
- Training: Scaled features

### Metrics
- Accuracy: 86.18% (without scaling), ~81% with scaling stable
- ROC-AUC: ~0.85
- Confusion Matrix: [[23,15],[4,81]] - TP 81, TN 23, FP 15, FN 4
- Classification Report: Precision Yes 84%, Recall Yes 95%, F1 ~90%
- Most Important Feature: Credit_History (highest coefficient)

### EDA Screenshots
- Approval Distribution: 69% Yes, 31% No
- Credit History vs Approval: Credit=1 -> 80% approved, Credit=0 -> 8% approved
- Education vs Approval: Graduate slightly higher
- Income vs Approval: Higher median for approved
- Correlation: ApplicantIncome-LoanAmount 0.5

### Application
- Framework: Streamlit
- UI Inputs: 11 fields (Gender, Married, Dependents, Education, Self_Employed, Incomes, LoanAmount, Term, Credit, Property)
- Output: Approval Probability (0-100%), Classification (Approved/Rejected)
- Test Cases:
    - Case 1: Credit 1, Income 6000, Urban -> 92% Approved (Pass)
    - Case 2: Credit 0, Income 2000 -> 8% Rejected (Pass)
    - Case 3: Semiurban, Credit 1 -> 78% Approved

### Limitations
- Small dataset (614 rows) - may not generalize
- Imbalanced (69/31)
- No feature engineering for TotalIncome
- Logistic Regression linear - can't capture complex patterns
- Missing external data (CIBIL score details)

