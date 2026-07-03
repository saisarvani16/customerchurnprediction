import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)
st.markdown("""
<style>

/* Professional Dark Background */
.stApp{
    background:
    radial-gradient(circle at top right,
    rgba(37,99,235,0.12),
    transparent 35%),

    radial-gradient(circle at bottom left,
    rgba(59,130,246,0.08),
    transparent 40%),

    linear-gradient(
        135deg,
        #0f172a 0%,
        #111827 50%,
        #0b1220 100%
    );

    background-attachment: fixed;
}

/* Remove default backgrounds */
[data-testid="stAppViewContainer"]{
    background: transparent;
}

[data-testid="stHeader"]{
    background: transparent;
}

.main{
    background: transparent;
}

</style>
""", unsafe_allow_html=True)
st.title("📊 Customer Churn Prediction")

st.write("Predict whether a customer is likely to churn.")

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("churn_model.pkl")
accuracy = joblib.load("accuracy.pkl")
feature_importance = joblib.load("feature_importance.pkl")
# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📊 Model Insights")

if st.sidebar.button("📈 Show Model Accuracy"):
    st.sidebar.metric(
        "Random Forest Accuracy",
        f"{accuracy*100:.2f}%"
    )

#st.sidebar.write("---")

if st.sidebar.button("📊 Show Important Features"):

    st.sidebar.subheader("Top 10 Features")

    fig, ax = plt.subplots(figsize=(5,4))

    top10 = feature_importance.head(10)

    ax.barh(
        top10["Feature"],
        top10["Importance"],
        color="skyblue"
    )

    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Features")
    ax.invert_yaxis()

    st.sidebar.pyplot(fig)
# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

customer_ids = data["customerID"]

# Remove customerID
data.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

# Fill missing values
data["TotalCharges"] = data["TotalCharges"].fillna(
    data["TotalCharges"].median()
)

# One Hot Encoding
data = pd.get_dummies(
    data,
    drop_first=True,
    dtype=int
)

# Features
X = data.drop("Churn_Yes", axis=1)

# -----------------------------
# Customer Selection
# -----------------------------
customer_id = st.selectbox(
    "Select Customer ID",
    customer_ids
)

st.write("---")

if st.button("Predict Churn"):
        # Find selected customer's index
    index = customer_ids[customer_ids == customer_id].index[0]

    # Get customer features
    customer = X.iloc[[index]]

    # Predict
    prediction = model.predict(customer)
    probability = model.predict_proba(customer)

    churn_prob = probability[0][1] * 100
    stay_prob = probability[0][0] * 100

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("🔴 Customer is likely to CHURN")
    else:
        st.success("🟢 Customer is likely to STAY")

    st.write(f"### Probability of Staying : {stay_prob:.2f}%")
    st.progress(int(stay_prob))

    st.write(f"### Probability of Churning : {churn_prob:.2f}%")
    st.progress(int(churn_prob))

    st.write("---")
    st.subheader("Customer Details")

    customer_details = pd.read_csv(
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    details = customer_details[
        customer_details["customerID"] == customer_id
    ]

    st.dataframe(details, use_container_width=True)
st.write("---")
