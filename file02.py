import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Find train file
csv_file = None
for root, dirs, files in os.walk("."):
    for f in files:
        if f.startswith("train") and f.endswith(".csv"):
            csv_file = os.path.join(root, f)

df = pd.read_csv(csv_file)

# --- Clean quickly ---
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

sns.set(style="whitegrid")

# 1. Approval analysis
plt.figure(figsize=(6,4))
sns.countplot(x='Loan_Status', data=df, palette='Set2')
plt.title('1. Loan Approval Distribution')
plt.savefig('1_approval_distribution.png')
plt.close()
print("Saved 1_approval_distribution.png")
print(df['Loan_Status'].value_counts())
print(df['Loan_Status'].value_counts(normalize=True).round(2))
print("\n")

# 2. Credit History vs Approval
plt.figure(figsize=(6,4))
pd.crosstab(df['Credit_History'], df['Loan_Status']).plot(kind='bar', stacked=False, color=['#FF9999','#66B2FF'])
plt.title('2. Credit History vs Loan Approval')
plt.xlabel('Credit_History (1=Good, 0=Bad)')
plt.savefig('2_credit_vs_approval.png')
plt.close()
print("Saved 2_credit_vs_approval.png")
print(pd.crosstab(df['Credit_History'], df['Loan_Status'], normalize='index').round(2))
print("\n")

# 3. Education vs Approval
plt.figure(figsize=(6,4))
sns.countplot(x='Education', hue='Loan_Status', data=df, palette='Set1')
plt.title('3. Education vs Loan Approval')
plt.savefig('3_education_vs_approval.png')
plt.close()
print("Saved 3_education_vs_approval.png\n")

# 4. Income distribution vs Approval
plt.figure(figsize=(6,4))
sns.boxplot(x='Loan_Status', y='ApplicantIncome', data=df)
plt.title('4. Applicant Income vs Loan Approval')
plt.savefig('4_income_vs_approval.png')
plt.close()
print("Saved 4_income_vs_approval.png\n")

# 5. Correlation heatmap - numerical features
plt.figure(figsize=(8,5))
corr = df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('5. Correlation / Relationship Analysis')
plt.savefig('5_correlation_heatmap.png')
plt.close()
print("Saved 5_correlation_heatmap.png\n")

print("ALL GRAPHS CREATED - Check folder left side for 5 PNG files")