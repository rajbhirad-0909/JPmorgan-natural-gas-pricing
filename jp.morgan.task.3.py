# ============================
#  TASK 3 – CREDIT RISK MODEL
#  JPMorgan Chase – Forage
# ============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ----------------------------
# LOAD DATA
# ----------------------------
file_path = r"C:\Users\HP\Desktop\Task 3 and 4_Loan_Data.csv"
df = pd.read_csv(file_path)

# Remove ID column (not predictive)
df = df.drop(columns=["customer_id"])

# Separate features & target
X = df.drop(columns=["default"])
y = df["default"]

# ----------------------------
# TRAIN / TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ----------------------------
# SCALE FEATURES
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# MODEL 1 — Logistic Regression
# ----------------------------
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train_scaled, y_train)

log_reg_pred = log_reg.predict_proba(X_test_scaled)[:, 1]
logreg_auc = roc_auc_score(y_test, log_reg_pred)

# ----------------------------
# MODEL 2 — Random Forest
# ----------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    random_state=42
)
rf.fit(X_train, y_train)

rf_pred = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_pred)

# ----------------------------
# CHOOSE BEST MODEL
# ----------------------------
if rf_auc > logreg_auc:
    best_model = rf
    model_name = "Random Forest"
else:
    best_model = log_reg
    model_name = "Logistic Regression"

print(f"Best model selected: {model_name}")
print(f"AUC Score: {max(rf_auc, logreg_auc):.4f}")

# =====================================================
# EXPECTED LOSS FUNCTION
# PD = Probability of Default from the model
# LGD = 1 - Recovery Rate  (given RR = 10%)
# Expected Loss = Exposure * PD * LGD
# =====================================================

RECOVERY_RATE = 0.10
LGD = 1 - RECOVERY_RATE

def predict_expected_loss(
    credit_lines_outstanding,
    loan_amt_outstanding,
    total_debt_outstanding,
    income,
    years_employed,
    fico_score,
    exposure=None
):
    """
    Returns PD (Probability of Default) and Expected Loss (EL)
    """
    
    # exposure is usually equal to loan_amt_outstanding
    if exposure is None:
        exposure = loan_amt_outstanding

    # build a single-row dataframe
    input_data = pd.DataFrame([[
        credit_lines_outstanding,
        loan_amt_outstanding,
        total_debt_outstanding,
        income,
        years_employed,
        fico_score
    ]], columns=X.columns)
    
    # scale if logistic regression
    if isinstance(best_model, LogisticRegression):
        input_scaled = scaler.transform(input_data)
        pd_value = best_model.predict_proba(input_scaled)[0, 1]
    else:
        pd_value = best_model.predict_proba(input_data)[0, 1]

    expected_loss = exposure * pd_value * LGD

    return pd_value, expected_loss

# ----------------------------
# TEST THE MODEL
# ----------------------------
sample_pd, sample_el = predict_expected_loss(
    credit_lines_outstanding=2,
    loan_amt_outstanding=5000,
    total_debt_outstanding=12000,
    income=55000,
    years_employed=3,
    fico_score=610
)

print("Sample PD:", round(sample_pd, 4))
print("Sample Expected Loss:", round(sample_el, 2))
