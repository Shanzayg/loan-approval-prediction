import pandas as pd, os, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

# Find file
csv_file = None
for root, dirs, files in os.walk("."):
    for f in files:
        if f.startswith("train") and f.endswith(".csv"):
            csv_file = os.path.join(root, f)
df = pd.read_csv(csv_file)

# Clean
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

# Scaling - fixes convergence warning
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:,1]

print("===== 1. ALL EVALUATION METRICS =====")
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"ROC-AUC: {auc:.4f}")

print("\n===== 2. CLASSIFICATION REPORT (Precision, Recall, F1) =====")
print(classification_report(y_test, y_pred, target_names=['No (0)','Yes (1)']))

print("\n===== 3. CONFUSION MATRIX =====")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

# Graphs for proof
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred No','Pred Yes'], yticklabels=['Actual No','Actual Yes'])
plt.title('Confusion Matrix')
plt.savefig('6_confusion_matrix.png')
plt.close()
print("\nSaved 6_confusion_matrix.png")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve')
plt.legend()
plt.savefig('6_roc_auc.png')
plt.close()
print("Saved 6_roc_auc.png")

print("\n===== 4. MODEL ANALYSIS =====")
feat_imp = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]}).sort_values('Coefficient', ascending=False)
print(feat_imp)