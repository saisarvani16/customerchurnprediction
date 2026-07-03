# ==============================
# Customer Churn Prediction
# Data Preprocessing & EDA
# ==============================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# ------------------------------
# Load Dataset
# ------------------------------
data = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ------------------------------
# Data Preprocessing
# ------------------------------

# Remove unnecessary column
# Save Customer IDs
customer_ids = data["customerID"].copy()

# Remove customerID from training data
data.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

# Fill missing values with median
data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

# ------------------------------
# Exploratory Data Analysis (EDA)
# ------------------------------

plt.figure(figsize=(15, 5))

# Churn Distribution
plt.subplot(1, 3, 1)
sns.countplot(x="Churn", data=data)
plt.title("Customer Churn Distribution")

# Gender vs Churn
plt.subplot(1, 3, 2)
sns.countplot(x="gender", hue="Churn", data=data)
plt.title("Gender vs Churn")

# Monthly Charges Distribution
plt.subplot(1, 3, 3)
sns.histplot(data["MonthlyCharges"], bins=20)
plt.title("Monthly Charges Distribution")
plt.xlabel("Monthly Charges")
plt.ylabel("Customer Count")

plt.tight_layout()
plt.show()

# ------------------------------
# Encode Categorical Columns
# ------------------------------
data = pd.get_dummies(data, drop_first=True, dtype=int)

# ------------------------------
# Separate Features and Target
# ------------------------------
X = data.drop("Churn_Yes", axis=1)
joblib.dump(list(X.columns), "feature_columns.pkl")
y = data["Churn_Yes"]


# Check Shapes
print("Features Shape :", X.shape)
print("Target Shape   :", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Features :", X_train.shape)
print("Testing Features  :", X_test.shape)
print("Training Target   :", y_train.shape)
print("Testing Target    :", y_test.shape)
from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train,y_train)
joblib.dump(rf_model, "churn_model.pkl")
y_pred = rf_model.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

joblib.dump(accuracy, "accuracy.pkl")
cm = confusion_matrix(y_test, y_pred)
print(cm)

joblib.dump(cm, "cm.pkl")
print(classification_report(y_test, y_pred))
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = (
    pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf_model.feature_importances_
    })
    .sort_values(by="Importance", ascending=False)
    .reset_index(drop=True)
)
# ------------------------------
# Predict Churn using Customer ID
# ------------------------------

# ------------------------------
# Predict Churn using Customer ID
# ------------------------------

print("\n======================================")
print("Model training completed successfully!")
print("Files Saved:")
print("✔ churn_model.pkl")
print("✔ accuracy.pkl")
print("✔ cm.pkl")
print("✔ feature_importance.pkl")
print("✔ feature_columns.pkl")
print("======================================")